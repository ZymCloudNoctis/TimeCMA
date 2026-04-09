#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 【分配给第 4 张卡】
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/mnt/cache/xlpr_sharedata/data_lhr/TimeCMA:$PYTHONPATH

data_path="Weather"
seq_len=96
batch_size=32
channel=64

log_path="./Results/${data_path}/"
mkdir -p $log_path

# ================= 任务 1: 预测未来 96 步 =================
pred_len=96
learning_rate=1e-3
e_layer=6
d_layer=2
dropout_n=0.1

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
echo "开始训练 Weather (pred_len=96)..."

/mnt/cache/xlpr_sharedata/data_lhr/envs/TimeCMA/bin/python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 21 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 20 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file

echo "Weather 96步 训练完成！"

# ================= 任务 2: 预测未来 336 步 =================
pred_len=336
learning_rate=1e-4
channel=32
e_layer=1
d_layer=2
dropout_n=0.1

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
echo "开始训练 Weather (pred_len=336)..."

/mnt/cache/xlpr_sharedata/data_lhr/envs/TimeCMA/bin/python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 21 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 100 \
  --seed 2024 \
  --channel $channel \
  --head 8 \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file