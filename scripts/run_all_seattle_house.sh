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
      --dataset seattle_house\
      --test-cutoff-date 2025-01-31 \
      --device cuda
  done
done

python scripts/make_report.py --runs runs/seattle_house/ --out runs/seattle_house/report

echo "All jobs finished."