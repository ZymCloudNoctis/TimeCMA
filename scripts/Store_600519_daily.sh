#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.." || exit
export PYTHONPATH=$PYTHONPATH:.

data_path="dataset/ETT-small/600519_enriched.csv"
root_path="."
num_nodes=25
freq="d"

# 优化建议：窗口缩短为 60 天（一个季度），更适合灵敏捕捉股市动态
input_len=60
output_len=5

mkdir -p ./Results/emb_logs/
divides=("train" "val" "test")

echo "开始生成灵敏版收益率嵌入 (seq=$input_len) ..."

for divide in "${divides[@]}"; do
    echo "正在处理分片: $divide ..."
    python -u storage/store_emb.py \
        --divide "$divide" \
        --root_path "$root_path" \
        --data_path "$data_path" \
        --num_nodes "$num_nodes" \
        --freq "$freq" \
        --input_len "$input_len" \
        --output_len "$output_len" \
        --batch_size 1
done
echo "嵌入生成完成！"
