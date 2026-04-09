#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/.." || exit
export PYTHONPATH=$PYTHONPATH:.

data_path="dataset/ETT-small/600519_enriched.csv"
root_path="."
num_nodes=25
freq="d"
seq_len=60   # 与生成脚本保持一致
pred_len=5

# --- 最终平衡版超参数 ---
batch_size=16           
learning_rate=0.00001   # 降低学习率，追求更稳健的收敛
channel=128             
e_layer=3
d_layer=2               
dropout_n=0.2           # 增加 Dropout，压制过拟合，提升泛化性
weight_decay=0.001      # 回调权重衰减
epochs=150
patience=40             # 早停耐心值

log_dir="./Results/600519_daily/"
mkdir -p "$log_dir"
log_file="${log_dir}sensitive_run.log"

echo "开始最终平衡版训练 (Returns, seq=${seq_len})..."
python -u train.py --data_path "$data_path" --root_path "$root_path" --num_nodes "$num_nodes" --seq_len "$seq_len" --pred_len "$pred_len" --batch_size "$batch_size" --freq "$freq" --learning_rate "$learning_rate" --channel "$channel" --e_layer "$e_layer" --d_layer "$d_layer" --dropout_n "$dropout_n" --weight_decay "$weight_decay" --epochs "$epochs" --es_patience "$patience" > "$log_file" 2>&1
