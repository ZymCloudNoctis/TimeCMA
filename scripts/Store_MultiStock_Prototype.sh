#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.." || exit
export PYTHONPATH=$PYTHONPATH:.

data_path="dataset/Ashare30/multistock_quotes.csv"
stock_pool_file="dataset/Ashare30/stock_pool.txt"
root_path="."
num_nodes=30
freq="d"
input_len=60
target_horizon=5

mkdir -p ./Results/emb_logs/
divides=("train" "val" "test")

echo "开始生成多股票版 TimeCMA 嵌入 (seq=$input_len, horizon=$target_horizon) ..."

for divide in "${divides[@]}"; do
    echo "正在处理分片: $divide ..."
    python -u storage/store_emb.py \
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
        --batch_size 1
done

echo "多股票嵌入生成完成！"
