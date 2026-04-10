#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.." || exit
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
python_bin="${PYTHON_BIN:-python3}"

data_path="${1:-${MULTISTOCK_DATA_PATH:-}}"
if [ -z "$data_path" ]; then
  data_path="dataset/HS300/raw_stock_data.csv"
fi

if [ ! -f "$data_path" ]; then
  echo "Market data csv not found: $data_path"
  echo "Usage: bash scripts/HS300_Snowball.sh /absolute/path/to/hs300_quotes.csv"
  echo "Or copy the file to dataset/HS300/raw_stock_data.csv."
  exit 1
fi

stock_pool_file="dataset/HS300/stock_pool.csv"
graph_file="dataset/HS300/hs300_news_cooccurrence_2025-01-01_2025-05-31.csv"
root_path="."
freq="d"
seq_len=60
target_horizon=5
num_nodes=300
start_date="2015-01-01"
end_date="2025-09-30"
trainval_start_date="2015-01-01"
trainval_end_date="2025-05-31"
test_start_date="2025-06-01"
test_end_date="2025-09-30"
val_ratio="0.1"

if [ ! -f "$stock_pool_file" ]; then
  echo "Stock pool file not found: $stock_pool_file"
  exit 1
fi

if [ ! -f "$graph_file" ]; then
  echo "Graph file not found: $graph_file"
  exit 1
fi

batch_size=16
learning_rate=0.0001
channel=128
e_layer=3
d_layer=2
dropout_n=0.2
weight_decay=0.001
epochs=100
patience=30

log_dir="./Results/HS300_Snowball/"
mkdir -p "$log_dir"
log_file="${log_dir}train.log"

echo "开始 HS300 多股票版 TimeCMA 训练 (seq=${seq_len}, horizon=${target_horizon})..."
echo "行情数据: $data_path"
echo "股票池: $stock_pool_file"
echo "静态图: $graph_file"
echo "全局时间窗口: $start_date ~ $end_date"
echo "训练/验证窗口: $trainval_start_date ~ $trainval_end_date"
echo "测试窗口: $test_start_date ~ $test_end_date"
echo "验证比例: $val_ratio"

"$python_bin" -u train.py \
  --task_name "multistock" \
  --data_path "$data_path" \
  --root_path "$root_path" \
  --stock_pool_file "$stock_pool_file" \
  --graph_file "$graph_file" \
  --num_nodes "$num_nodes" \
  --seq_len "$seq_len" \
  --target_horizon "$target_horizon" \
  --start_date "$start_date" \
  --end_date "$end_date" \
  --trainval_start_date "$trainval_start_date" \
  --trainval_end_date "$trainval_end_date" \
  --test_start_date "$test_start_date" \
  --test_end_date "$test_end_date" \
  --val_ratio "$val_ratio" \
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
