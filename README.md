<div align="center">
  <h2><b> (AAAI'25) TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment </b></h2>
</div>

This repository contains the code for our AAAI 2025 [paper](https://arxiv.org/abs/2406.01638), where we propose an intuitive yet effective framework for MTSF via cross-modality alignment.

> If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚:

```bibtex
@inproceedings{liu2025timecma,
  title={{TimeCMA}: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment},
  author={Liu, Chenxi and Xu, Qianxiong and Miao, Hao and Yang, Sun and Zhang, Lingzheng and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={AAAI},
  year={2025}
}
```

## Abstract
Multivariate time series forecasting (MTSF) aims to learn temporal dynamics among variables to forecast future time series. Existing statistical and deep learning-based methods suffer from limited learnable parameters and small-scale training data. Recently, large language models (LLMs) combining time series with textual prompts have achieved promising performance in MTSF. However, we discovered that current LLM-based solutions fall short in learning *disentangled* embeddings. We introduce TimeCMA, an intuitive yet effective framework for MTSF via cross-modality alignment. Specifically, we present a dual-modality encoding with two branches: the time series encoding branch extracts *disentangled yet weak* time series embeddings, and the LLM-empowered encoding branch wraps the same time series with text as prompts to obtain *entangled yet robust* prompt embeddings. As a result, such a cross-modality alignment retrieves *both disentangled and robust* time series embeddings, ``the best of two worlds'', from the prompt embeddings based on time series and prompt modality similarities. As another key design, to reduce the computational costs from time series with their length textual prompts, we design an effective prompt to encourage the most essential temporal information to be encapsulated in the last token: only the last token is passed to downstream prediction. We further store the last token embeddings to accelerate inference speed. Extensive experiments on eight real datasets demonstrate that TimeCMA outperforms state-of-the-arts.

<p align="center">
  <img width="900" alt="image" src="https://github.com/user-attachments/assets/f7359297-5781-4f09-b7b6-aa82f0df817d" />
</p>

## Dependencies

* Python 3.11
* PyTorch 2.1.2
* CUDA 12.1
* torchvision 0.8.0

```bash
> conda env create -f env_{ubuntu,windows}.yaml
```

## Datasets
Datasets can be obtained from [TimesNet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) and [TFB](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view).

## Usages
* ### Last token embedding storage

```bash
bash Store_{data_name}.sh
```

* ### Train and inference
   
```bash
bash {data_name}.sh
```

* ### Co-occurrence graph construction

Build a COGRASP-style co-occurrence graph from the multivariate variables in a TimeCMA dataset:

```bash
python build_graph.py \
  --root_path . \
  --data_path dataset/ETT-small/600519_enriched.csv \
  --output_path dataset/ETT-small/600519_enriched_cooccurrence_train.csv \
  --split train \
  --seq_len 60 \
  --pred_len 5
```

The stock `600519` training script now supports `--auto_build_graph` and will automatically create or reuse the graph file before training.

## Multi-Stock Prototype

This repo now includes a multi-stock TimeCMA path where nodes are stocks instead of factor columns.

### Expected market data format

Provide a long-form CSV with at least these canonical fields:

```text
date,stock_code,open,high,low,close,vol,amount
```

The loader also accepts common aliases used by A-share daily bars, including:

```text
ts_code/code -> stock_code
volume -> vol
turnover -> amount
trade_date -> date
```

Provide a stock pool file (`.txt` or `.csv`) whose order is the single source of truth for:

* time-series node order
* LLM embedding node order
* graph matrix row/column order

Provide your Snowball / COGRASP co-occurrence matrix as a square CSV whose row and column labels match the stock pool codes.

### Generate multi-stock prompt embeddings

```bash
bash scripts/Store_MultiStock_Prototype.sh
```

### Train the multi-stock model

```bash
bash scripts/MultiStock_Prototype.sh
```

The multi-stock training path predicts future `target_horizon`-day returns per stock and reports `MSE`, `MAE`, and cross-sectional `RankIC`.

### HS300 + Snowball static graph

This repo now includes the HS300 files needed by the multi-stock pipeline directly inside `TimeCMA`:

* `dataset/HS300/all_stocks_complete_data.csv`
* `dataset/HS300/stock_pool.csv`
* `dataset/HS300/hs300_news_cooccurrence_2024-01-01_2025-05-31.csv`

The default split policy is:

* train/validation sample range: `2024-01-01` to `2025-05-31`
* test sample range: `2025-06-01` to `2025-09-30`
* validation split: the last `20%` of train/validation samples in chronological order

Run:

```bash
bash scripts/Store_HS300_Snowball.sh
bash scripts/HS300_Snowball.sh
```

If you want to point to another market CSV, pass it as the first argument:

```bash
bash scripts/Store_HS300_Snowball.sh /absolute/path/to/hs300_quotes.csv
bash scripts/HS300_Snowball.sh /absolute/path/to/hs300_quotes.csv
```

The default static graph is the Snowball news co-occurrence matrix built from `2024-01-01` through `2025-05-31`, and the training/validation window is aligned to that period.
