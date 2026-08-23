"""Per-instance case library: for a set of already-trained model checkpoints,
compute each model's error on every (region, lookback/forecast window)
instance across the full train+val+test span, and record the best model per
instance.

Each run directory is loaded independently via
:func:`housets_bench.experiments.run_loader.load_run` (its own saved
``config.yaml`` + ``checkpoint.pt``) — models can use different transform
pipelines (e.g. PCA for some ML baselines), so each gets its own bundle built
from its own config rather than sharing one; only the dataset/window/split
*shape* is required to match across runs, which is validated up front.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from housets_bench.bundles.datatypes import ProcBundle
from housets_bench.experiments.run_loader import load_run
from housets_bench.metrics.evaluator import invert_to_raw
from housets_bench.models.base import BaseForecaster

MODEL_CATEGORY: Dict[str, str] = {
    # spatial_temporal (GNN)
    "gcn_tcn": "spatial_temporal",
    "graph_wavenet": "spatial_temporal",
    "stgcn": "spatial_temporal",
    "stsgcn": "spatial_temporal",
    "stllm_plus": "spatial_temporal",
    "dcrnn": "spatial_temporal",
    "stgformer": "spatial_temporal",
    "d2stgnn": "spatial_temporal",
    "cast": "spatial_temporal",
    "stexplainer": "spatial_temporal",
    "aist": "spatial_temporal",
    "st_hhol": "spatial_temporal",
    # DL
    "rnn": "DL",
    "lstm": "DL",
    "dlinear": "DL",
    "timemixer": "DL",
    "patchtst": "DL",
    "informer": "DL",
    "autoformer": "DL",
    "fedformer": "DL",
    "itransformer": "DL",
    "gpt4ts": "DL",
    "timellm": "DL",
    # foundation
    "chronos_zero": "foundation",
    "chronos_ft": "foundation",
    "chronos_full_ft": "foundation",
    "chronos2_zero": "foundation",
    "chronos2_ft": "foundation",
    "chronos2_full_ft": "foundation",
    "timesfm_zero": "foundation",
    "timesfm_xreg_zero": "foundation",
    "timesfm_xreg_ft": "foundation",
    "timesfm_ft": "foundation",
    "timesfm_full_ft": "foundation",
    # other: statistical / classical ML — not one of the user's 3 requested
    # categories (DL / foundation / spatial_temporal), kept so these don't
    # get silently mislabeled if included in the model list.
    "ar_univariate": "other",
    "ardl": "other",
    "arima": "other",
    "var": "other",
    "var_ms": "other",
    "rf": "other",
    "rf_pca": "other",
    "xgb": "other",
    "xgb_pca": "other",
}

CASE_LIBRARY_COLUMNS = [
    "region_id",
    "lookback_start",
    "forecast_start",
    "forecast_end",
    "model_best",
    "model_best_mae",
    "model_best_rmse",
    "model_best_category",
]

DETAIL_COLUMNS = [
    "model",
    "model_category",
    "split",
    "region_id",
    "lookback_start",
    "forecast_start",
    "forecast_end",
    "step_ahead",
    "target_date",
    "y_true_raw",
    "y_pred_raw",
    "abs_error_raw",
]

LONG_COLUMNS = [
    "model",
    "model_category",
    "split",
    "region_id",
    "lookback_start",
    "forecast_start",
    "forecast_end",
    "mae",
    "rmse",
]

_DATA_KEYS = ["path", "id_col", "time_col", "target_col", "feature_cols", "n_zip"]
_WINDOW_KEYS = ["seq_len", "label_len", "pred_len"]
_SPLIT_KEYS = ["train_ratio", "val_ratio", "test_start_date"]


def model_category(model_name: str) -> str:
    return MODEL_CATEGORY.get(model_name, "other")


def _validate_run_consistency(
    ref_run_dir: Path, run_dir: Path, ref_cfg: Dict[str, Any], cfg: Dict[str, Any]
) -> None:
    mismatches: List[str] = []
    ref_data, data = (ref_cfg.get("data", {}) or {}), (cfg.get("data", {}) or {})
    ref_window, window = (ref_cfg.get("window", {}) or {}), (cfg.get("window", {}) or {})
    ref_split, split = (ref_cfg.get("split", {}) or {}), (cfg.get("split", {}) or {})

    for k in _DATA_KEYS:
        if data.get(k) != ref_data.get(k):
            mismatches.append(f"data.{k}: {ref_data.get(k)!r} vs {data.get(k)!r}")
    for k in _WINDOW_KEYS:
        if window.get(k) != ref_window.get(k):
            mismatches.append(f"window.{k}: {ref_window.get(k)!r} vs {window.get(k)!r}")
    for k in _SPLIT_KEYS:
        if split.get(k) != ref_split.get(k):
            mismatches.append(f"split.{k}: {ref_split.get(k)!r} vs {split.get(k)!r}")

    if mismatches:
        raise ValueError(
            f"Run {run_dir} is not comparable to {ref_run_dir} — a per-instance case library "
            "requires every run to share the same dataset, window shape, and split boundaries:\n  "
            + "\n  ".join(mismatches)
        )


def _iter_instance_forecasts(
    model: BaseForecaster,
    bundle: ProcBundle,
    split: str,
    *,
    device: Optional[torch.device],
    max_batches: Optional[int],
) -> Iterator[Tuple[Dict[str, Any], np.ndarray, np.ndarray]]:
    """Yield (meta, y_true_raw[H], y_pred_raw[H]) for every instance in ``split``.

    ``meta`` includes ``forecast_dates`` (length-H list) alongside the usual
    region_id/lookback_start/forecast_start/forecast_end, so callers can build
    the full per-step detail table without re-running the model.
    """
    pred_len = int(bundle.raw.spec.pred_len)
    seq_len = int(bundle.raw.spec.seq_len)
    dates = bundle.raw.aligned.dates
    zipcodes = bundle.raw.aligned.zipcodes
    n_dates = len(dates)

    def _meta(region_id: str, t0: int, t_pred_start: int) -> Dict[str, Any]:
        forecast_dates = [dates[min(t_pred_start + h, n_dates - 1)] for h in range(pred_len)]
        return {
            "region_id": region_id,
            "lookback_start": dates[t0],
            "forecast_start": forecast_dates[0],
            "forecast_end": forecast_dates[-1],
            "forecast_dates": forecast_dates,
        }

    graph_dls = getattr(model, "_graph_dataloaders", None)
    is_gnn = bool(graph_dls and split in graph_dls)

    if is_gnn:
        dl = graph_dls[split]
        time_anchors = dl.dataset.time_anchors
        n_nodes = len(zipcodes)
        item_idx = 0
        for bi, batch in enumerate(dl):
            if max_batches is not None and bi >= max_batches:
                break
            y_true = batch["y"][:, -pred_len:, :]
            y_pred = model.predict_batch(batch, bundle=bundle, device=device)
            t_raw, p_raw = invert_to_raw(bundle, y_true, y_pred)  # [B*N, H]

            n_items = int(batch["x"].shape[0])
            for b in range(n_items):
                t0 = time_anchors[item_idx]
                item_idx += 1
                t_pred_start = t0 + seq_len
                for n in range(n_nodes):
                    row = b * n_nodes + n
                    yield _meta(zipcodes[n], t0, t_pred_start), t_raw[row], p_raw[row]
    else:
        dl = bundle.dataloaders[split]
        for bi, batch in enumerate(dl):
            if max_batches is not None and bi >= max_batches:
                break
            y_true = batch["y"][:, -pred_len:, :]
            y_pred = model.predict_batch(batch, bundle=bundle, device=device)
            t_raw, p_raw = invert_to_raw(bundle, y_true, y_pred)  # [B, H]

            for i, meta in enumerate(batch["meta"]):
                t0 = int(meta["t0"])
                t_pred_start = int(meta["t_pred_start"])
                yield _meta(meta["zipcode"], t0, t_pred_start), t_raw[i], p_raw[i]


def _aggregate_long(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-step detail table to one row per (model, instance): mae/rmse."""
    if detail_df.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    group_cols = ["model", "model_category", "split", "region_id", "lookback_start", "forecast_start", "forecast_end"]
    grouped = detail_df.groupby(group_cols)["abs_error_raw"]
    mae = grouped.mean()
    rmse = grouped.apply(lambda s: float(np.sqrt(np.mean(np.square(s)))))
    return pd.DataFrame({"mae": mae, "rmse": rmse}).reset_index()[LONG_COLUMNS]


def _pick_best(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=CASE_LIBRARY_COLUMNS)

    group_cols = ["region_id", "lookback_start", "forecast_start", "forecast_end"]
    idx = long_df.groupby(group_cols)["mae"].idxmin()
    best = long_df.loc[idx].rename(
        columns={
            "model": "model_best",
            "mae": "model_best_mae",
            "rmse": "model_best_rmse",
            "model_category": "model_best_category",
        }
    )
    return best[CASE_LIBRARY_COLUMNS].sort_values(group_cols).reset_index(drop=True)


def build_case_library(
    run_dirs: List[Union[str, Path]],
    *,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the per-instance case library across ``run_dirs``.

    Returns ``(case_library_df, long_df, detail_df)``:
      - ``detail_df``: the finest-grained table — one row per (model,
        instance, horizon step), with the actual forecasted and true values
        (raw target units) and that step's absolute error. Computed first,
        directly from each model's predictions, before any summarization.
      - ``long_df``: one row per (model, instance) — mae/rmse aggregated over
        the horizon, derived from ``detail_df``.
      - ``case_library_df``: the requested 8-column table — one row per
        instance, the winning model only — derived from ``long_df``.
    """
    run_dirs = [Path(r) for r in run_dirs]
    if not run_dirs:
        raise ValueError("run_dirs must be non-empty")

    detail_records: List[Dict[str, Any]] = []
    ref_run_dir: Optional[Path] = None
    ref_cfg: Optional[Dict[str, Any]] = None

    for run_dir in run_dirs:
        model, bundle, cfg = load_run(
            run_dir, device=device, cfg_overrides={"window": {"test_stride": 1}}
        )
        if ref_cfg is None:
            ref_run_dir, ref_cfg = run_dir, cfg
        else:
            _validate_run_consistency(ref_run_dir, run_dir, ref_cfg, cfg)

        model_name = str((cfg.get("model", {}) or {}).get("name"))
        category = model_category(model_name)
        if verbose:
            print(f"[case_library] scoring {model_name} ({category}) from {run_dir} ...")

        n_instances = 0
        for split in ("train", "val", "test"):
            for meta, y_true_raw, y_pred_raw in _iter_instance_forecasts(
                model, bundle, split, device=device, max_batches=max_batches
            ):
                forecast_dates = meta["forecast_dates"]
                for h, target_date in enumerate(forecast_dates):
                    yt = float(y_true_raw[h])
                    yp = float(y_pred_raw[h])
                    detail_records.append(
                        {
                            "model": model_name,
                            "model_category": category,
                            "split": split,
                            "region_id": meta["region_id"],
                            "lookback_start": meta["lookback_start"],
                            "forecast_start": meta["forecast_start"],
                            "forecast_end": meta["forecast_end"],
                            "step_ahead": h + 1,
                            "target_date": target_date,
                            "y_true_raw": yt,
                            "y_pred_raw": yp,
                            "abs_error_raw": abs(yp - yt),
                        }
                    )
                n_instances += 1
        if verbose:
            print(f"[case_library]   {n_instances} instances scored")

    detail_df = pd.DataFrame.from_records(detail_records)
    if not detail_df.empty:
        detail_df = detail_df[DETAIL_COLUMNS]
    long_df = _aggregate_long(detail_df)
    case_library_df = _pick_best(long_df)

    if verbose:
        n_models = long_df["model"].nunique() if not long_df.empty else 0
        n_instances_total = len(case_library_df)
        print(f"[case_library] {n_instances_total} instances, {n_models} models compared")
        if not case_library_df.empty:
            print("[case_library] win counts:")
            print(case_library_df["model_best"].value_counts().to_string())

    return case_library_df, long_df, detail_df
