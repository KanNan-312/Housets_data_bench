#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "dlinear"
  "patchtst"
  "itransformer"
  "timemixer"
  "timellm"
  "gpt4ts"
  "stllm_plus"
  "chronos2_ft"
  "timesfm_ft"
  "chronos2_zero"
  "timesfm_zero"
  "stgcn"
  "stsgcn"
  "graph_wavenet"
  "dcrnn"
  "d2stgnn"
  "stgformer"
  "aist"
  "st_hhol"
  "stexplainer"
  "cast"
)

DATASETS=(
  "chicago_crime"
  "seattle_house"
)

for DATASET in "${DATASETS[@]}"; do
  echo "=== Dataset: ${DATASET} ==="

  for MODEL in "${MODELS[@]}"; do
    echo "Launching dataset=${DATASET}, model=${MODEL}"
    python scripts/run_one.py \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --device cuda
  done

  echo "Generating report for ${DATASET}..."
  python scripts/make_report.py \
    --runs "runs/${DATASET}/" \
    --out "runs/${DATASET}/report"
done

echo "All jobs finished."