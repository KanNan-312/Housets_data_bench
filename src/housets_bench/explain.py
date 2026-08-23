"""On-demand neighbor-contribution explainability for spatiotemporal (GNN) models.

Model-agnostic occlusion: zero one node's input features (across the whole
lookback window) in the model's own processed feature space — under this
benchmark's z-score transform, zeroing is approximately "replace with that
feature's average", not an implausible perturbation — rerun the forward
pass, and measure how much the target region's forecast changes. This works
identically for every spatiotemporal model in the registry (gcn_tcn,
graph_wavenet, stgcn, stsgcn, stllm_plus, dcrnn, stgformer, d2stgnn, cast,
stexplainer, aist, st_hhol) since it only calls the public
``model.predict_batch(...)`` API — no per-architecture attention plumbing.

Only called for one instance at a time (not baked into the case library,
which would need one forward pass per neighbor per instance).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from housets_bench.data.schema import normalize_zipcode
from housets_bench.experiments.run_loader import load_run
from housets_bench.graph.loader import load_graph
from housets_bench.metrics.evaluator import invert_to_raw
from housets_bench.models.gnn.gnn_forecaster import GNNForecasterBase


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_raw_horizon(bundle, y_proc_row: np.ndarray) -> np.ndarray:
    """[H, Dy] processed -> [H] raw target values."""
    arr = y_proc_row[None, :, :]
    _, p_raw = invert_to_raw(bundle, arr, arr)
    return p_raw[0]


def explain_neighbors(
    run_dir: Union[str, Path],
    *,
    region_id: str,
    lookback_start: str,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Neighbor-contribution breakdown for one (region, lookback_start) instance.

    Returns a DataFrame with columns ``neighbor_region_id, contribution_pct,
    delta_mae_raw, is_self``, sorted by contribution descending.
    """
    model, bundle, cfg = load_run(run_dir, device=device)

    if not isinstance(model, GNNForecasterBase):
        raise ValueError(
            f"explain_neighbors only applies to spatiotemporal/GNN models "
            f"(got model={cfg.get('model', {}).get('name')!r}, which is not a GNNForecasterBase)"
        )

    graph_cfg = bundle.raw.graph
    if not graph_cfg.path:
        raise ValueError("This run's dataset config has no graph.path set — cannot resolve neighbors")

    zipcodes = bundle.raw.aligned.zipcodes
    geo = load_graph(graph_cfg.path, zipcodes)

    target_norm = normalize_zipcode(region_id)
    zip_norm = [normalize_zipcode(z) for z in zipcodes]
    if target_norm not in zip_norm:
        raise ValueError(f"region_id={region_id!r} not found in this run's regions")
    target_idx = zip_norm.index(target_norm)

    dates = bundle.raw.aligned.dates
    target_date = pd.Timestamp(lookback_start)
    date_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    if target_date not in date_idx:
        raise ValueError(
            f"lookback_start={lookback_start!r} does not match any date in this dataset "
            f"(range: {dates[0]} .. {dates[-1]})"
        )
    t0 = date_idx[target_date]

    seq_len = int(bundle.raw.spec.seq_len)
    if t0 + seq_len > len(dates):
        raise ValueError(f"lookback_start={lookback_start!r} leaves too few future steps for seq_len={seq_len}")

    # neighbor set: any node connected to target_idx in either direction, excluding self-loops
    src, dst = np.asarray(geo.edge_index[0]), np.asarray(geo.edge_index[1])
    neighbor_idxs = sorted(
        (set(dst[src == target_idx].tolist()) | set(src[dst == target_idx].tolist())) - {target_idx}
    )

    name_to_idx = {n: i for i, n in enumerate(bundle.aligned_proc.schema.continuous_cols)}
    x_idx = [name_to_idx[c] for c in bundle.x_cols]
    values = bundle.aligned_proc.values  # [N, T, D]
    x_np = values[:, t0 : t0 + seq_len, :][:, :, x_idx]  # [N, L, Dx]
    x_base = torch.tensor(x_np, dtype=torch.float32).permute(1, 0, 2).unsqueeze(0)  # [1, L, N, Dx]

    def _target_forecast_raw(x: torch.Tensor) -> np.ndarray:
        y = model.predict_batch({"x": x}, bundle=bundle, device=device)  # [N, pred_len, Dy]
        y_target = _to_numpy(y[target_idx])  # [pred_len, Dy]
        return _to_raw_horizon(bundle, y_target)

    baseline_raw = _target_forecast_raw(x_base)

    rows = []
    for node_idx, label, is_self in [(target_idx, "self", True)] + [
        (n, zipcodes[n], False) for n in neighbor_idxs
    ]:
        x_occ = x_base.clone()
        x_occ[:, :, node_idx, :] = 0.0
        occ_raw = _target_forecast_raw(x_occ)
        delta = float(np.mean(np.abs(occ_raw - baseline_raw)))
        rows.append({"neighbor_region_id": label, "delta_mae_raw": delta, "is_self": is_self})

    df = pd.DataFrame(rows)
    total = df["delta_mae_raw"].sum()
    if total > 0:
        df["contribution_pct"] = 100.0 * df["delta_mae_raw"] / total
    else:
        df["contribution_pct"] = 100.0 / max(len(df), 1)

    return df[["neighbor_region_id", "contribution_pct", "delta_mae_raw", "is_self"]].sort_values(
        "contribution_pct", ascending=False
    ).reset_index(drop=True)
