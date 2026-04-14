#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.." || exit
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
python_bin="${PYTHON_BIN:-python3}"

data_path="${1:-${MULTISTOCK_DATA_PATH:-}}"
if [ -z "$data_path" ]; then
  data_path="dataset/HS300/all_stocks_complete_data.csv"
fi

if [ ! -f "$data_path" ]; then
  echo "Market data csv not found: $data_path"
  echo "Usage: bash scripts/HS300_Snowball.sh /absolute/path/to/hs300_quotes.csv"
  echo "Or copy the file to dataset/HS300/all_stocks_complete_data.csv."
  exit 1
fi

canonical_stock_pool_file="dataset/HS300/stock_pool.csv"
canonical_graph_file="dataset/HS300/hs300_news_cooccurrence_2024-01-01_2025-05-31.csv"
runtime_dir="dataset/HS300/runtime"
stock_pool_file="${runtime_dir}/stock_pool_runtime.csv"
graph_file="${runtime_dir}/graph_runtime.csv"
root_path="."
freq="d"
seq_len=60
target_horizon=5
num_nodes=300
start_date="2024-01-01"
end_date="2025-09-30"
trainval_start_date="2024-01-01"
trainval_end_date="2025-05-31"
test_start_date="2025-06-01"
test_end_date="2025-09-30"
val_ratio="0.2"
target_winsorize_lower="0.01"
target_winsorize_upper="0.99"

if [ ! -f "$canonical_stock_pool_file" ]; then
  echo "Canonical stock pool file not found: $canonical_stock_pool_file"
  exit 1
fi

if [ ! -f "$canonical_graph_file" ]; then
  echo "Canonical graph file not found: $canonical_graph_file"
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
graph_weight_transform="log1p"
graph_top_k=10

log_dir="./Results/HS300_Snowball/"
mkdir -p "$log_dir"
log_file="${log_dir}train.log"

echo "开始 HS300 多股票版 TimeCMA 训练 (seq=${seq_len}, horizon=${target_horizon})..."
echo "行情数据: $data_path"
echo "原始股票池: $canonical_stock_pool_file"
echo "原始静态图: $canonical_graph_file"
echo "全局时间窗口: $start_date ~ $end_date"
echo "训练/验证窗口: $trainval_start_date ~ $trainval_end_date"
echo "测试窗口: $test_start_date ~ $test_end_date"
echo "验证比例: $val_ratio"
echo "训练标签 winsorize: $target_winsorize_lower ~ $target_winsorize_upper"
echo "图权重变换: $graph_weight_transform"
echo "图 top-k: $graph_top_k"

"$python_bin" -u build_runtime_assets.py \
  --market_data "$data_path" \
  --stock_pool_file "$canonical_stock_pool_file" \
  --graph_file "$canonical_graph_file" \
  --output_stock_pool "$stock_pool_file" \
  --output_graph "$graph_file"

echo "运行时股票池: $stock_pool_file"
echo "运行时静态图: $graph_file"
echo "训练日志: $log_file"

"$python_bin" -u train.py \
  --task_name "multistock" \
  --data_path "$data_path" \
  --root_path "$root_path" \
  --stock_pool_file "$stock_pool_file" \
  --graph_file "$graph_file" \
  --graph_weight_transform "$graph_weight_transform" \
  --graph_top_k "$graph_top_k" \
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
  --target_winsorize_lower "$target_winsorize_lower" \
  --target_winsorize_upper "$target_winsorize_upper" \
  --batch_size "$batch_size" \
  --freq "$freq" \
  --learning_rate "$learning_rate" \
  --channel "$channel" \
  --e_layer "$e_layer" \
  --d_layer "$d_layer" \
  --dropout_n "$dropout_n" \
  --weight_decay "$weight_decay" \
  --epochs "$epochs" \
  --es_patience "$patience" 2>&1 | tee "$log_file"
