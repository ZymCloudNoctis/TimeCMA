#!/bin/bash
set -euo pipefail
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
  echo "Usage: bash scripts/Store_HS300_Snowball.sh /absolute/path/to/hs300_quotes.csv"
  echo "Or copy the file to dataset/HS300/all_stocks_complete_data.csv."
  exit 1
fi

canonical_stock_pool_file="dataset/HS300/stock_pool.csv"
canonical_graph_file="dataset/HS300/hs300_news_cooccurrence_2024-01-01_2025-05-31.csv"
runtime_dir="dataset/HS300/runtime"
stock_pool_file="${runtime_dir}/stock_pool_runtime.csv"
graph_file="${runtime_dir}/graph_runtime.csv"
root_path="."
num_nodes=300
freq="d"
input_len=60
target_horizon=5
start_date="2024-01-01"
end_date="2025-09-30"
trainval_start_date="2024-01-01"
trainval_end_date="2025-05-31"
test_start_date="2025-06-01"
test_end_date="2025-09-30"
val_ratio="0.2"

if [ ! -f "$canonical_stock_pool_file" ]; then
  echo "Canonical stock pool file not found: $canonical_stock_pool_file"
  exit 1
fi

if [ ! -f "$canonical_graph_file" ]; then
  echo "Canonical graph file not found: $canonical_graph_file"
  exit 1
fi

mkdir -p ./Results/emb_logs/
divides=("train" "val" "test")

echo "开始生成 HS300 多股票版 TimeCMA 嵌入 (seq=$input_len, horizon=$target_horizon) ..."
echo "行情数据: $data_path"
echo "原始股票池: $canonical_stock_pool_file"
echo "原始静态图: $canonical_graph_file"
echo "全局时间窗口: $start_date ~ $end_date"
echo "训练/验证窗口: $trainval_start_date ~ $trainval_end_date"
echo "测试窗口: $test_start_date ~ $test_end_date"
echo "验证比例: $val_ratio"

"$python_bin" -u build_runtime_assets.py \
  --market_data "$data_path" \
  --stock_pool_file "$canonical_stock_pool_file" \
  --graph_file "$canonical_graph_file" \
  --output_stock_pool "$stock_pool_file" \
  --output_graph "$graph_file"

echo "运行时股票池: $stock_pool_file"
echo "运行时静态图: $graph_file"

for divide in "${divides[@]}"; do
  echo "正在处理分片: $divide ..."
  "$python_bin" -u storage/store_emb.py \
    --task_name "multistock" \
    --divide "$divide" \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --stock_pool_file "$stock_pool_file" \
    --num_nodes "$num_nodes" \
    --freq "$freq" \
    --input_len "$input_len" \
    --output_len "$target_horizon" \
    --target_horizon "$target_horizon" \
    --start_date "$start_date" \
    --end_date "$end_date" \
    --trainval_start_date "$trainval_start_date" \
    --trainval_end_date "$trainval_end_date" \
    --test_start_date "$test_start_date" \
    --test_end_date "$test_end_date" \
    --val_ratio "$val_ratio" \
    --batch_size 1
done

echo "HS300 多股票嵌入生成完成！"
