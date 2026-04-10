#!/bin/bash
cd "$(dirname "$0")/.." || exit
export PYTHONPATH=$PYTHONPATH:.

python -u build_graph.py \
  --root_path "." \
  --data_path "dataset/ETT-small/600519_enriched.csv" \
  --output_path "dataset/ETT-small/600519_enriched_cooccurrence_train.csv" \
  --split "train" \
  --seq_len 60 \
  --pred_len 5 \
  --event_threshold 0.5 \
  --min_weight 1
