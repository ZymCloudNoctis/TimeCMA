#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$(dirname "$0")/.." || exit
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

python_bin="${PYTHON_BIN:-python3}"
market_data="${1:-dataset/HS300/all_stocks_complete_data.csv}"
snowball_csv="${2:-dataset/Snowball/snowball_posts_clean_2011_2025.csv.gz}"

if [ ! -f "$market_data" ]; then
  echo "Market data csv not found: $market_data"
  exit 1
fi

if [ ! -f "$snowball_csv" ]; then
  echo "Snowball cleaned csv not found: $snowball_csv"
  exit 1
fi

echo "开始运行 HS300 滚动窗口实验..."
echo "行情数据: $market_data"
echo "雪球数据: $snowball_csv"
echo "方法: no_graph static dynamic6m"
echo "测试月份: 2025-04 ~ 2025-09"
echo "图权重变换: log1p + top-k(10)"
echo "动态图时间衰减: exp, half-life=60 days"
echo "训练标签 winsorize: 1% ~ 99%"

"$python_bin" -u run_rolling_experiments.py \
  --market_data "$market_data" \
  --snowball_csv "$snowball_csv" \
  --methods no_graph static dynamic6m \
  --test_month_start 2025-04 \
  --test_month_end 2025-09 \
  --graph_weight_transform log1p \
  --graph_top_k 10 \
  --dynamic_time_decay exp \
  --dynamic_decay_half_life_days 60 \
  --target_winsorize_lower 0.01 \
  --target_winsorize_upper 0.99
