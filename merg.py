import pandas as pd
import numpy as np
import os

# 定义路径
base_path = 'dataset/ETT-small/600519_with_index.csv'
tech_path = 'dataset/ETT-small/technical_factors_600519.csv'
output_path = 'dataset/ETT-small/600519_enriched.csv'

# 加载原始数据
df_base = pd.read_csv(base_path)
# 将日期转换为统一格式 YYYYMMDD 用于连接，然后再转回 YYYY-MM-DD
df_base['join_date'] = df_base['date'].str.replace('-', '').astype(int)

# 加载技术因子数据
# 预读一部分看看列名
df_tech_full = pd.read_csv(tech_path)

# 选择需要的列 (bfq 表示不复权数据，因为 base 数据看起来也是不复权的)
tech_cols = [
    'trade_date',
    'turnover_rate', 
    'macd_bfq', 'macd_dea_bfq', 'macd_dif_bfq',
    'rsi_bfq_6', 'rsi_bfq_12', 'rsi_bfq_24',
    'kdj_k_bfq', 'kdj_d_bfq',
    'boll_upper_bfq', 'boll_mid_bfq', 'boll_lower_bfq'
]

df_tech = df_tech_full[tech_cols]

# 合并
df_merged = pd.merge(df_base, df_tech, left_on='join_date', right_on='trade_date', how='left')

# 删除临时连接列
df_merged = df_merged.drop(columns=['join_date', 'trade_date'])

# 处理缺失值 (技术指标在最初几天可能有空值)
df_merged = df_merged.fillna(method='ffill').fillna(method='bfill')

# 确保 OT (收盘价) 依然在最后一列
cols = list(df_merged.columns)
cols.remove('OT')
cols.append('OT')
df_merged = df_merged[cols]

# 保存
df_merged.to_csv(output_path, index=False)
print(f"Successfully merged data. New shape: {df_merged.shape}")
print(f"Columns: {df_merged.columns.tolist()}")
