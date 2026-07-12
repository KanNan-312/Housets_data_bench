"""Evaluate a previously trained model without re-training.

Usage
-----
    python scripts/eval_one.py --run-dir runs/dlinear__multivariate__w6_h3
    python scripts/eval_one.py --run-dir runs/... --splits test --device cuda
    python scripts/eval_one.py --run-dir runs/... --splits val test --max-eval-batches 50

The script reads config.yaml and checkpoint.pt from the run directory, rebuilds
the data bundle, loads the checkpoint, runs evaluation on the requested splits,
and writes eval_metrics.json back into the same run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from housets_bench.experiments.artifacts import save_json
from housets_bench.experiments.sweep import build_bundle_from_cfg, apply_hparams
from housets_bench.data.io import AlignedData, load_aligned
from housets_bench.metrics.evaluator import evaluate_forecaster
from housets_bench.models.registry import get as get_model
from housets_bench.utils.config import load_yaml, resolve_relpaths

import housets_bench.models  # ensure all models are registered


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    p.add_argument("--run-dir", type=str, required=True,
                   help="Path to a run directory containing config.yaml and checkpoint.pt")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["train", "val", "test"],
                   help="Which splits to evaluate (default: val test)")
    p.add_argument("--device", type=str, default=None,
                   help="Override device (e.g. cuda, cpu)")
    p.add_argument("--max-eval-batches", type=int, default=None,
                   help="Cap number of eval batches (0 = all)")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path (default: <run-dir>/eval_metrics.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoint.pt"

    if not config_path.exists():
        raise SystemExit(f"config.yaml not found in {run_dir}")

    cfg: Dict[str, Any] = load_yaml(config_path)
    resolve_relpaths(cfg, root=REPO_ROOT)

    # device
    run_cfg = cfg.get("run", {}) or {}
    device_str = args.device or run_cfg.get("device", "cpu")
    dev = torch.device(device_str)

    max_eval = args.max_eval_batches
    if max_eval is not None and max_eval <= 0:
        max_eval = None

    # load data
    data_cfg = cfg.get("data", {}) or {}
    aligned = load_aligned(
        data_cfg.get("path"),
        target_col=str(data_cfg.get("target_col", "price")),
        impute=bool(data_cfg.get("impute", True)),
    )
    n_zip = int(data_cfg.get("n_zip", 0) or 0)
    if n_zip > 0 and aligned.n_zip > n_zip:
        zips = aligned.zipcodes[:n_zip]
        zip_mask = np.isin(np.array(aligned.zipcodes), np.array(zips))
        aligned = AlignedData(
            zipcodes=list(np.array(aligned.zipcodes)[zip_mask]),
            dates=aligned.dates,
            values=aligned.values[zip_mask],
            time_marks=aligned.time_marks,
            schema=aligned.schema,
        )

    bundle = build_bundle_from_cfg(aligned=aligned, cfg=cfg)

    # instantiate model + apply hparams
    model_cfg = cfg.get("model", {}) or {}
    model_name = str(model_cfg.get("name"))
    model = get_model(model_name)
    apply_hparams(model, model_cfg.get("hparams", {}) or {})

    # load checkpoint (required for DL models; no-op for non-DL)
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_checkpoint(checkpoint_path, device=dev)
    else:
        print(f"[warn] No checkpoint.pt found in {run_dir}. Evaluating without loading weights.")

    # evaluate
    results: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "model": model_name,
        "task": (cfg.get("task", {}) or {}).get("name"),
        "window": (cfg.get("window", {}) or {}).get("name"),
        "pipeline": bundle.pipeline.summary(),
    }

    for split in args.splits:
        print(f"Evaluating {split} ...")
        res = evaluate_forecaster(model, bundle, split=split, device=dev, max_batches=max_eval)
        results[split] = asdict(res)

    out_path = Path(args.out) if args.out else run_dir / "eval_metrics.json"
    save_json(out_path, results)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
