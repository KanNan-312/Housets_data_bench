from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from housets_bench.bundles.datatypes import ProcBundle
from housets_bench.models.base import BaseForecaster


def _to_numpy(x: Any) -> np.ndarray:
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


@dataclass
class EvalResult:
    rmse: float
    mape: float
    n_points: int
    metric_space: str = "log"   # "log" or "original"


class StreamingEvaluator:
    """Accumulates predictions and ground-truth across batches and computes final metrics.

    The pipeline is always **fully inverted** to raw price space first. The two metric
    spaces then differ only in how RMSE is computed on top of those raw prices:

    Args:
        metric_space:
            ``"log"``      – log-RMSE: ``RMSE(log1p(p_raw), log1p(t_raw))``.
                             Comparable across models regardless of internal pipeline
                             (fair for LLMs that predict directly on original scale).
            ``"original"`` – RMSE in raw dollar price space.
            MAPE is always in raw price space regardless of this flag.
    """

    def __init__(self, bundle: ProcBundle, *, eps: float = 1e-8, metric_space: str = "log") -> None:
        self.bundle = bundle
        self.eps = float(eps)

        metric_space = metric_space.lower()
        if metric_space not in ("log", "original"):
            raise ValueError("metric_space must be 'log' or 'original'")
        self.metric_space = metric_space

        # mapping from y_cols (model output space) -> processed full feature indices
        proc_names = list(bundle.aligned_proc.schema.continuous_cols)
        name_to_idx = {n: i for i, n in enumerate(proc_names)}
        self.y_idx_full = [name_to_idx[n] for n in bundle.y_cols]

        self.d_proc = int(bundle.aligned_proc.values.shape[-1])
        self.raw_target_index = int(bundle.raw_target_index)

        self._sse = 0.0
        self._sum_ape = 0.0
        self._n = 0

    def update(self, y_true_proc: Any, y_pred_proc: Any) -> None:
        yt = _to_numpy(y_true_proc).astype(np.float32)
        yp = _to_numpy(y_pred_proc).astype(np.float32)

        if yt.shape != yp.shape:
            raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
        if yt.ndim != 3:
            raise ValueError(f"Expected [B,H,Dy], got {yt.shape}")

        B, H, Dy = yt.shape

        # Embed into full processed feature dimension
        full_t = np.zeros((B, H, self.d_proc), dtype=np.float32)
        full_p = np.zeros((B, H, self.d_proc), dtype=np.float32)
        full_t[:, :, self.y_idx_full] = yt
        full_p[:, :, self.y_idx_full] = yp

        # Always fully invert the pipeline → raw price space.
        # This is pipeline-agnostic: works correctly regardless of whether zscore,
        # clip, or any other stage is enabled, and makes the metric fair for models
        # that predict directly on original scale (e.g. LLMs).
        t_raw_full = self.bundle.pipeline.inverse(full_t, keep_log=False)
        p_raw_full = self.bundle.pipeline.inverse(full_p, keep_log=False)
        t_raw = t_raw_full[:, :, self.raw_target_index].astype(np.float64)
        p_raw = p_raw_full[:, :, self.raw_target_index].astype(np.float64)

        if self.metric_space == "log":
            # log-RMSE: apply log1p to raw prices then compute RMSE.
            # Clamp to ≥0 before log1p to guard against tiny negatives from float precision.
            t_space = np.log1p(np.maximum(t_raw, 0.0))
            p_space = np.log1p(np.maximum(p_raw, 0.0))
        else:
            # RMSE directly in raw dollar price space
            t_space = t_raw
            p_space = p_raw

        diff = p_space - t_space
        self._sse += float(np.sum(diff * diff))
        self._n += int(diff.size)

        # MAPE always in raw price space
        denom = np.maximum(np.abs(t_raw), self.eps)
        self._sum_ape += float(np.sum(np.abs(p_raw - t_raw) / denom))

    def compute(self) -> EvalResult:
        if self._n == 0:
            return EvalResult(rmse=float("nan"), mape=float("nan"), n_points=0, metric_space=self.metric_space)
        rmse = float(np.sqrt(self._sse / self._n))
        mape = float(self._sum_ape / self._n)
        return EvalResult(rmse=rmse, mape=mape, n_points=self._n, metric_space=self.metric_space)


@torch.no_grad()
def evaluate_forecaster(
    model: BaseForecaster,
    bundle: ProcBundle,
    *,
    split: str = "test",
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    metric_space: str = "log",
) -> EvalResult:
    """Evaluate a forecaster on the given split.

    Args:
        metric_space: Passed to :class:`StreamingEvaluator`.
            ``"log"`` evaluates RMSE in log-price space (default, scale-normalised).
            ``"original"`` evaluates RMSE in raw price space (dollar scale).
    """
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of: train/val/test")

    dl = bundle.dataloaders[split]
    evaluator = StreamingEvaluator(bundle, metric_space=metric_space)

    pred_len = int(bundle.raw.spec.pred_len)
    total = min(len(dl), max_batches) if max_batches is not None else len(dl)
    dl_bar = tqdm(dl, desc=f"  eval/{split}", total=total, leave=False, unit="bt")
    for bi, batch in enumerate(dl_bar):
        if max_batches is not None and bi >= max_batches:
            dl_bar.close()
            break

        # batch['y'] includes [label_len+pred_len]; we evaluate only the forecast horizon
        y_true = batch["y"][:, -pred_len:, :]

        y_pred = model.predict_batch(batch, bundle=bundle, device=device)

        evaluator.update(y_true, y_pred)

    return evaluator.compute()
