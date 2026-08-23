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

# seq_len / label_len / pred_len for w12_h12; other combos: w6_h12 (6/6/12), w6_h6 (6/3/6), w12_h6 (12/6/6)
SEQ_LEN=12
LABEL_LEN=12
PRED_LEN=12

for MODEL in "${MODELS[@]}"; do
  echo "Launching model=${MODEL}, seq_len=${SEQ_LEN}, pred_len=${PRED_LEN}"
  python scripts/run_one.py \
    --model "${MODEL}" \
    --seq-len "${SEQ_LEN}" \
    --label-len "${LABEL_LEN}" \
    --pred-len "${PRED_LEN}" \
    --dataset chicago_crime \
    --test-cutoff-date 2025-01-31 \
    --device cuda
done

python scripts/make_report.py --runs runs/chicago_crime/ --out runs/chicago_crime/report

echo "All jobs finished."