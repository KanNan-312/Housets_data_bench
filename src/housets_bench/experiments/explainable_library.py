"""Builds the 3-model-family explainable case library for one dataset.

Different from :mod:`housets_bench.case_library` (which ranks many models
against each other): this always uses exactly **one** model per family —
1. a univariate model (e.g. PatchTST, TimesFM) — point forecasts over every instance
2. a multivariate model (e.g. Chronos-2, iTransformer) — point forecasts over
   every instance, plus grouped-Shapley feature importance
   (:mod:`housets_bench.shap_occlusion`) for a sampled subset of instances
3. STExplainer — point forecasts over every instance, plus its native
   node/time/feature explanation (:meth:`STExplainerForecaster.explain`) for
   every instance (a single extra forward pass per window, not the expensive
   ``2^N`` SHAP sweep)

and its output is forecasts + explanations, not a "best model" ranking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from housets_bench.case_library import DETAIL_COLUMNS, _iter_instance_forecasts, _join_values, model_category
from housets_bench.experiments.run_loader import load_run
from housets_bench.models.gnn.stexplainer import STExplainerForecaster
from housets_bench.shap_occlusion import shapley_feature_importance

FEATURE_IMPORTANCE_COLUMNS = ["region_id", "lookback_start", "forecast_start", "group", "shap_value", "contribution_pct"]
STEXPLAINER_EXPLANATION_COLUMNS = [
    "region_id", "lookback_start", "forecast_start", "forecast_end", "axis", "object_id", "importance", "contribution_pct"
]


def _forecast_detail_rows(
    run_dir: Union[str, Path], *, device: Optional[torch.device], max_batches: Optional[int]
) -> Tuple[List[Dict[str, Any]], object, object, Dict[str, Any]]:
    """Point-forecast detail rows for one run, over every instance (all splits)."""
    model, bundle, cfg = load_run(run_dir, device=device, cfg_overrides={"window": {"test_stride": 1}})
    model_name = str((cfg.get("model", {}) or {}).get("name"))
    category = model_category(model_name)

    records: List[Dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for meta, y_true_raw, y_pred_raw in _iter_instance_forecasts(
            model, bundle, split, device=device, max_batches=max_batches
        ):
            diff = y_pred_raw - y_true_raw
            mse = float(np.mean(np.square(diff)))
            records.append(
                {
                    "model": model_name,
                    "model_category": category,
                    "split": split,
                    "region_id": meta["region_id"],
                    "lookback_start": meta["lookback_start"],
                    "forecast_start": meta["forecast_start"],
                    "forecast_end": meta["forecast_end"],
                    "mse": mse,
                    "mae": float(np.mean(np.abs(diff))),
                    "rmse": float(np.sqrt(mse)),
                    "y_true_raw": _join_values(y_true_raw),
                    "y_pred_raw": _join_values(y_pred_raw),
                }
            )
    return records, model, bundle, cfg


def _sample_test_instances(
    model, bundle, *, device: Optional[torch.device], n: int
) -> List[Tuple[str, Any, Any]]:
    """First ``n`` distinct (region_id, lookback_start) instances from the test split, time order."""
    seen: List[Tuple[str, Any, Any]] = []
    for meta, _y_true, _y_pred in _iter_instance_forecasts(model, bundle, "test", device=device, max_batches=None):
        key = (meta["region_id"], meta["lookback_start"], meta["forecast_start"])
        if key not in seen:
            seen.append(key)
        if len(seen) >= n:
            break
    return seen


def _feature_importance_rows(
    run_dir: Union[str, Path],
    model,
    bundle,
    *,
    device: Optional[torch.device],
    n_instances: int,
    groups: Optional[Dict[str, List[str]]],
    n_temporal_groups: int,
) -> List[Dict[str, Any]]:
    instances = _sample_test_instances(model, bundle, device=device, n=n_instances)
    records: List[Dict[str, Any]] = []
    for region_id, lookback_start, forecast_start in instances:
        shap_df = shapley_feature_importance(
            run_dir,
            region_id=region_id,
            lookback_start=lookback_start,
            groups=groups,
            n_temporal_groups=n_temporal_groups,
            device=device,
        )
        for _, row in shap_df.iterrows():
            records.append(
                {
                    "region_id": region_id,
                    "lookback_start": lookback_start,
                    "forecast_start": forecast_start,
                    "group": row["group"],
                    "shap_value": float(row["shap_value"]),
                    "contribution_pct": float(row["contribution_pct"]),
                }
            )
    return records


def _stexplainer_explanation_rows(
    run_dir: Union[str, Path], *, device: Optional[torch.device], max_batches: Optional[int]
) -> List[Dict[str, Any]]:
    model, bundle, cfg = load_run(run_dir, device=device, cfg_overrides={"window": {"test_stride": 1}})
    if not isinstance(model, STExplainerForecaster):
        raise ValueError(
            f"--stexplainer-model run must use a stexplainer model "
            f"(got {(cfg.get('model', {}) or {}).get('name')!r})"
        )

    zipcodes = bundle.raw.aligned.zipcodes
    x_cols = list(bundle.x_cols)
    dates = bundle.raw.aligned.dates
    seq_len = int(bundle.raw.spec.seq_len)
    pred_len = int(bundle.raw.spec.pred_len)
    n_dates = len(dates)

    graph_dls = model._graph_dataloaders
    records: List[Dict[str, Any]] = []

    for split in ("train", "val", "test"):
        dl = graph_dls[split]
        time_anchors = dl.dataset.time_anchors
        item_idx = 0
        for bi, batch in enumerate(dl):
            if max_batches is not None and bi >= max_batches:
                break
            n_items = int(batch["x"].shape[0])
            for b in range(n_items):
                t0 = time_anchors[item_idx]
                item_idx += 1
                t_pred_start = t0 + seq_len
                end_idx = min(t_pred_start + pred_len - 1, n_dates - 1)
                lookback_start, forecast_start, forecast_end = dates[t0], dates[t_pred_start], dates[end_idx]

                x_single = batch["x"][b : b + 1]
                exp = model.explain(x_single)  # one extra forward pass per window
                att_spat = exp["att_spat"]  # [N, N]
                time_imp = exp["time_importance"]  # [L]
                feat_imp = exp["feature_importance"]  # [Dx]

                for ti, region_id in enumerate(zipcodes):
                    node_imp = (att_spat[ti, :] + att_spat[:, ti]) / 2.0
                    for tj, neighbor_id in enumerate(zipcodes):
                        if tj == ti:
                            continue
                        records.append(
                            _explanation_row(
                                region_id, lookback_start, forecast_start, forecast_end,
                                "node", neighbor_id, node_imp[tj],
                            )
                        )
                    for t_idx in range(seq_len):
                        records.append(
                            _explanation_row(
                                region_id, lookback_start, forecast_start, forecast_end,
                                "time", str(t_idx), time_imp[t_idx],
                            )
                        )
                    for f_idx, col in enumerate(x_cols):
                        records.append(
                            _explanation_row(
                                region_id, lookback_start, forecast_start, forecast_end,
                                "feature", col, feat_imp[f_idx],
                            )
                        )
    return records


def _explanation_row(region_id, lookback_start, forecast_start, forecast_end, axis, object_id, importance) -> Dict[str, Any]:
    return {
        "region_id": region_id,
        "lookback_start": lookback_start,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "axis": axis,
        "object_id": object_id,
        "importance": float(importance),
    }


def _normalize_stexplainer_pct(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["region_id", "lookback_start", "axis"]
    total = df.groupby(group_cols)["importance"].transform(lambda s: s.abs().sum())
    df["contribution_pct"] = np.where(total > 0, 100.0 * df["importance"].abs() / total, 100.0 / df.groupby(group_cols)["importance"].transform("size"))
    return df


def build_explainable_case_library(
    *,
    univariate_run_dir: Union[str, Path],
    multivariate_run_dir: Union[str, Path],
    stexplainer_run_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    explain_n_instances: int = 20,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    n_temporal_groups: int = 1,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (forecasts_detail_df, feature_importance_df, stexplainer_explanation_df)."""
    all_records: List[Dict[str, Any]] = []

    if verbose:
        print(f"[explainable_library] univariate forecasts from {univariate_run_dir} ...")
    uni_records, _uni_model, _uni_bundle, _uni_cfg = _forecast_detail_rows(
        univariate_run_dir, device=device, max_batches=max_batches
    )
    all_records += uni_records

    if verbose:
        print(f"[explainable_library] multivariate forecasts from {multivariate_run_dir} ...")
    mv_records, mv_model, mv_bundle, _mv_cfg = _forecast_detail_rows(
        multivariate_run_dir, device=device, max_batches=max_batches
    )
    all_records += mv_records

    if verbose:
        print(f"[explainable_library] stexplainer forecasts from {stexplainer_run_dir} ...")
    ste_records, _ste_model, _ste_bundle, _ste_cfg = _forecast_detail_rows(
        stexplainer_run_dir, device=device, max_batches=max_batches
    )
    all_records += ste_records

    forecasts_detail_df = pd.DataFrame.from_records(all_records)
    if not forecasts_detail_df.empty:
        forecasts_detail_df = forecasts_detail_df[DETAIL_COLUMNS]

    if verbose:
        print(
            f"[explainable_library] grouped-SHAP feature importance for "
            f"{explain_n_instances} test instances of the multivariate model ..."
        )
    fi_records = _feature_importance_rows(
        multivariate_run_dir, mv_model, mv_bundle,
        device=device, n_instances=explain_n_instances,
        groups=feature_groups, n_temporal_groups=n_temporal_groups,
    )
    feature_importance_df = pd.DataFrame.from_records(fi_records)
    if not feature_importance_df.empty:
        feature_importance_df = feature_importance_df[FEATURE_IMPORTANCE_COLUMNS]

    if verbose:
        print(f"[explainable_library] STExplainer node/time/feature explanation for every instance ...")
    ste_exp_records = _stexplainer_explanation_rows(stexplainer_run_dir, device=device, max_batches=max_batches)
    stexplainer_explanation_df = pd.DataFrame.from_records(ste_exp_records)
    if not stexplainer_explanation_df.empty:
        stexplainer_explanation_df = _normalize_stexplainer_pct(stexplainer_explanation_df)
        stexplainer_explanation_df = stexplainer_explanation_df[STEXPLAINER_EXPLANATION_COLUMNS]

    if verbose:
        print(
            f"[explainable_library] done: {len(forecasts_detail_df)} forecast rows, "
            f"{len(feature_importance_df)} SHAP rows, {len(stexplainer_explanation_df)} explanation rows"
        )

    return forecasts_detail_df, feature_importance_df, stexplainer_explanation_df
