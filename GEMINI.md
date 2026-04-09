# TimeCMA: LLM-Empowered Multivariate Time Series Forecasting

TimeCMA is a framework for Multivariate Time Series Forecasting (MTSF) that leverages Large Language Models (LLMs) through cross-modality alignment. It combines time series encoding with LLM-empowered prompts to produce robust and disentangled embeddings for accurate forecasting.

## Project Overview

- **Purpose:** To improve MTSF by aligning time series modality with textual prompt modality using LLMs (e.g., GPT-2).
- **Core Technology:**
  - **Dual-Modality Encoding:** Features a time series branch (disentangled but weak) and an LLM branch (entangled but robust).
  - **Cross-Modality Alignment:** Retrieves the "best of both worlds" by aligning these two modalities.
  - **Efficiency:** Uses a "last token" strategy to compress long textual prompts and pre-stores LLM embeddings to accelerate training/inference.
- **Main Technologies:** Python 3.11, PyTorch 2.1.2, CUDA 12.1, Transformers (Hugging Face).

## Project Structure

- `train.py`: The main entry point for model training and evaluation.
- `models/TimeCMA.py`: Contains the core architecture of the `Dual` model.
- `storage/`:
  - `store_emb.py`: Extracts and saves LLM embeddings as `.h5` files to avoid re-computing them during training.
  - `gen_prompt_emb.py`: Logic for generating LLM prompts from time series data.
- `data_provider/`: Custom dataset and data loader implementations for various datasets (ETTh1, ETTh2, ETTm1, etc.).
- `scripts/`: Shell scripts to automate the embedding storage and training processes.
- `layers/`: Custom neural network components (Attention, Positional Encoding, etc.).
- `utils/`: Metrics, tools, and time feature extraction utilities.
- `Embeddings/`: Storage directory for pre-computed LLM embeddings.
- `Results/`: Logs and performance metrics.

## Building and Running

### Prerequisites

- **Environment:** Python 3.11, PyTorch 2.1.2, CUDA 12.1.
- **Setup:** (Based on README)
  ```bash
  conda env create -f env_ubuntu.yaml # or env_windows.yaml
  ```
- **PYTHONPATH:** The scripts expect `PYTHONPATH` to include the project root.

### Workflow

1. **Embedding Storage (Mandatory Pre-processing):**
   Pre-compute the LLM embeddings for a specific dataset (e.g., ETTh1).
   ```bash
   bash scripts/Store_ETTh1.sh
   ```
   This populates the `Embeddings/ETTh1/` directory with `.h5` files.

2. **Training and Inference:**
   Run the training script for the dataset.
   ```bash
   bash scripts/ETTh1.sh
   ```

## Development Conventions

- **Model Configuration:** Parameters like `seq_len`, `pred_len`, `channel`, and `dropout_n` are passed via CLI arguments to `train.py`.
- **Logging:** Training progress is logged to the `Results/` directory.
- **Reproducibility:** A seed (default `2024`) is used for random number generation in `train.py`.
- **Data Scaling:** The project uses standard scaling for time series data, typically handled within the custom dataset classes in `data_provider/`.
- **Hardcoded Paths:** Some scripts in `scripts/` may contain hardcoded paths (e.g., `/mnt/cache/...`). These should be updated to local paths if running in a different environment.

## Key Files to Reference

- `models/TimeCMA.py`: Architecture details.
- `train.py`: Training loop and argument parsing.
- `data_provider/data_loader_emb.py`: Data loading logic during training (using pre-stored embeddings).
- `data_provider/data_loader_save.py`: Data loading logic for embedding extraction.
- `storage/gen_prompt_emb.py`: Prompt engineering and LLM interaction.
