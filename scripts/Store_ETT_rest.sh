#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/mnt/cache/xlpr_sharedata/data_lhr/TimeCMA:$PYTHONPATH

mkdir -p ./Results/emb_logs/

# 只跑剩下的三个
data_paths=("ETTm1" "ETTm2" "ETTh2")
divides=("train" "val" "test")
num_nodes=7
input_len=96
output_len=96

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    log_file="./Results/emb_logs/${data_path}_${divide}.log"
    echo "正在提取 ${data_path} 的 ${divide} 特征..."
    
    # 替换为绝对路径，并去掉了 nohup 和 &
    /mnt/cache/xlpr_sharedata/data_lhr/envs/TimeCMA/bin/python storage/store_emb.py \
      --divide $divide \
      --data_path $data_path \
      --num_nodes $num_nodes \
      --input_len $input_len \
      --output_len $output_len > $log_file
      
    echo "${data_path} - ${divide} 特征提取完成！"
  done
done