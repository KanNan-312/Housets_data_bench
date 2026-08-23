from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from housets_bench.metrics.reporting import collect_runs, pivot_metric



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, default=str(REPO_ROOT / "runs" / "section4_sweep"))
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "runs" / "section4_sweep" / "reports"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_runs(run_root)
    if df.empty:
        raise SystemExit(f"No runs found under {run_root}")

    df.to_csv(out_dir / "metrics_long.csv", index=False)

    tasks = sorted(df["task"].dropna().unique().tolist())
    for task in tasks:
        for metric in ("log_rmse", "rmse", "mape", "mae", "log_mae"):
            piv = pivot_metric(df, task=task, split="test", metric=metric)
            if not piv.empty:
                piv.to_csv(out_dir / f"pivot_test_{metric}_{task}.csv")

    print(f"Wrote report to: {out_dir}")


if __name__ == "__main__":
    main()
