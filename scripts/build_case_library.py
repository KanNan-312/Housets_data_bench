"""Build a per-instance case library from a set of already-trained runs.

For each model listed (or every run found under --runs-root), reload its
checkpoint, score every (region, lookback/forecast window) instance across
the full train+val+test span in raw target units, and record which model
wins each instance.

Usage
-----
    python scripts/build_case_library.py \
      --runs-root runs/dc_house \
      --models dlinear patchtst gcn_tcn stgformer \
      --out runs/dc_house/case_library.csv

Omit --models to include every run found directly under --runs-root (each
subdirectory containing a config.yaml + checkpoint.pt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from housets_bench.case_library import build_case_library
from housets_bench.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=str, required=True,
                   help="directory containing one subdirectory per run (config.yaml + checkpoint.pt)")
    p.add_argument("--models", nargs="+", default=None,
                   help="model names to include (matches config.yaml's model.name); omit for all runs found")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu (default: each run's own config)")
    p.add_argument("--max-eval-batches", type=int, default=None, help="cap batches per split per run (debugging)")
    p.add_argument("--out", type=str, required=True, help="output CSV path for the case library")
    p.add_argument("--out-long", type=str, default=None,
                   help="optional CSV path for the full per-model long table (every model x every instance)")
    return p.parse_args()


def _discover_run_dirs(runs_root: Path, models: Optional[List[str]]) -> List[Path]:
    candidates = [
        child for child in sorted(runs_root.iterdir())
        if child.is_dir() and (child / "config.yaml").exists() and (child / "checkpoint.pt").exists()
    ]
    if not candidates:
        raise SystemExit(f"No run directories with config.yaml + checkpoint.pt found under {runs_root}")

    if models is None:
        return candidates

    wanted = set(models)
    selected = []
    for run_dir in candidates:
        cfg = load_yaml(run_dir / "config.yaml")
        name = str((cfg.get("model", {}) or {}).get("name", ""))
        if name in wanted:
            selected.append(run_dir)

    found_names = {str((load_yaml(r / "config.yaml").get("model", {}) or {}).get("name", "")) for r in candidates}
    missing = wanted - found_names
    if missing:
        raise SystemExit(f"No run found under {runs_root} for model(s): {sorted(missing)}")
    return selected


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"--runs-root not found or not a directory: {runs_root}")

    run_dirs = _discover_run_dirs(runs_root, args.models)
    print(f"Found {len(run_dirs)} run(s):")
    for r in run_dirs:
        print(f"  {r}")

    device = torch.device(args.device) if args.device else None
    case_library_df, long_df = build_case_library(
        run_dirs, device=device, max_batches=args.max_eval_batches, verbose=True
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    case_library_df.to_csv(out_path, index=False)
    print(f"\nSaved case library ({len(case_library_df)} instances) -> {out_path}")

    if args.out_long:
        out_long_path = Path(args.out_long)
        out_long_path.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(out_long_path, index=False)
        print(f"Saved long table ({len(long_df)} rows) -> {out_long_path}")


if __name__ == "__main__":
    main()
