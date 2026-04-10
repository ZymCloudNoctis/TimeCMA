#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.." || exit
export PYTHONPATH=$PYTHONPATH:.

data_path="dataset/Ashare30/multistock_quotes.csv"
stock_pool_file="dataset/Ashare30/stock_pool.txt"
graph_file="dataset/Ashare30/snowball_cooccurrence.csv"
root_path="."
freq="d"
seq_len=60
target_horizon=5
num_nodes=30

batch_size=16
learning_rate=0.0001
channel=128
e_layer=3
d_layer=2
dropout_n=0.2
weight_decay=0.001
epochs=100
patience=30

log_dir="./Results/MultiStock_Prototype/"
mkdir -p "$log_dir"
log_file="${log_dir}train.log"

echo "开始多股票版 TimeCMA 训练 (seq=${seq_len}, horizon=${target_horizon})..."
python -u train.py \
  --task_name "multistock" \
  --data_path "$data_path" \
  --root_path "$root_path" \
  --stock_pool_file "$stock_pool_file" \
  --graph_file "$graph_file" \
  --num_nodes "$num_nodes" \
  --seq_len "$seq_len" \
  --target_horizon "$target_horizon" \
  --batch_size "$batch_size" \
  --freq "$freq" \
  --learning_rate "$learning_rate" \
  --channel "$channel" \
  --e_layer "$e_layer" \
  --d_layer "$d_layer" \
  --dropout_n "$dropout_n" \
  --weight_decay "$weight_decay" \
  --epochs "$epochs" \
  --es_patience "$patience" > "$log_file" 2>&1
