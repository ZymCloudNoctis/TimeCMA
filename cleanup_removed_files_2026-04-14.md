# 清理记录

- 时间：2026-04-14
- 清理标准：仅保留当前 HS300 多股票、雪球共现图、滚动窗口实验链路所需的代码、主数据和文档。

## 已删除的旧文件与目录

### 1. 旧单股票与原型脚本

- `scripts/600519_daily.sh`
- `scripts/Build_600519_graph.sh`
- `scripts/Store_600519_daily.sh`
- `scripts/MultiStock_Prototype.sh`
- `scripts/Store_MultiStock_Prototype.sh`

### 2. 原始 TimeCMA benchmark 脚本

- `scripts/ECL.sh`
- `scripts/ETTh1.sh`
- `scripts/ETTh1_backup.sh`
- `scripts/ETTh2.sh`
- `scripts/ETTm1.sh`
- `scripts/ETTm2.sh`
- `scripts/FRED.sh`
- `scripts/ILI.sh`
- `scripts/Store_ECL.sh`
- `scripts/Store_ETT.sh`
- `scripts/Store_ETT_rest.sh`
- `scripts/Store_ETTh1.sh`
- `scripts/Store_FRED.sh`
- `scripts/Store_ILI.sh`
- `scripts/Store_Weather.sh`
- `scripts/Weather.sh`

### 3. 已废弃的辅助代码与临时产物

- `GEMINI.md`
- `build_graph.py`
- `merg.py`
- `summary_results.py`
- `preds.npy`
- `trues.npy`
- `data_provider/data_loader_temp.py`
- `data_provider/data_loader_emb.py`
- `data_provider/data_loader_save.py`

### 4. 已废弃的数据与中间文件

- `dataset/ETT-small/`
- `dataset/HS300/raw_stock_data.csv`
- `dataset/HS300/hs300_news_cooccurrence_2025-01-01_2025-05-31.csv`
- `dataset/Snowball/snowball_posts_clean_2011_2025_preview.csv`
- `dataset/Snowball/snowball_posts_clean_2011_2025_stats.csv`
- `dataset/Snowball/snowball_posts_clean_2011_2025_summary.json`

### 5. 可重建的运行产物与缓存

- `Embeddings/`
- `Results/`
- `scripts/Results/`
- 所有 `__pycache__/`
- 所有 `.DS_Store`

## 保留内容

以下内容仍保留，用于当前论文实验复现：

- `train.py`
- `run_rolling_experiments.py`
- `build_runtime_assets.py`
- `build_snowball_cooccurrence_matrix.py`
- `clean_snowball_xlsx.py`
- `models/`、`layers/`、`utils/`、`storage/`
- `scripts/HS300_Snowball.sh`
- `scripts/Store_HS300_Snowball.sh`
- `scripts/Rolling_HS300_Experiments.sh`
- `dataset/HS300/all_stocks_complete_data.csv`
- `dataset/HS300/stock_pool.csv`
- `dataset/HS300/hs300_news_cooccurrence_2024-01-01_2025-05-31.csv`
- `dataset/Snowball/snowball_posts_clean_2011_2025.csv.gz`
