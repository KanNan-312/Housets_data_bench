#!/usr/bin/env bash

MODELS=(
  "dlinear"
  "patchtst"
  "itransformer"
  "xgb"
  "chronos2_ft"
  "timesfm_ft"
  "timemixer"
  "stgcn"
  "stsgcn"
  "timellm"
  "graph_wavenet"
  "gpt4ts"
  "stllm_plus"
  "gcn_tcn"
)

# WINDOWS=("w12_h12" "w6_h12", "w6_h6", "w12_h6")
WINDOWS=("w12_h12")

for MODEL in "${MODELS[@]}"; do
  for WINDOW in "${WINDOWS[@]}"; do
    echo "Launching model=${MODEL}, window=${WINDOW}"
    python scripts/run_one.py \
      --model "${MODEL}" \
      --window "${WINDOW}" \
      --dataset chicago_crime \
      --test-cutoff-date 2025-01-31 \
      --device cuda
  done
done

python scripts/make_report.py --runs runs/chicago_crime/ --out runs/chicago_crime/report

echo "All jobs finished."