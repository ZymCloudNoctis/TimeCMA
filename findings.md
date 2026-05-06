# Findings

## 2026-05-01

- `train.py` currently supports only `TimeCMA` via `models.TimeCMA.Dual`.
- Dataset support already exists for:
  - `MultiStockEmbeddingDataset`: returns `(x, y, x_mark, y_mark, embeddings)`
  - `MultiStockSaveDataset`: returns `(x, y, x_mark, y_mark)` and does not require embeddings.
- Rolling evaluation already computes:
  - `MSE`, `MAE`
  - `IC`, `ICIR`
  - `RankIC`, `RankICIR`
- Existing rolling runner currently assumes:
  - embedding generation is needed before training every method
  - graph construction is conditional by method
- Baselines can be integrated fairly by:
  - reusing the same date windows and stock pool runtime assets
  - skipping embedding generation for non-TimeCMA methods
  - using a separate baseline training script rather than overloading `train.py`

## Baseline Design Direction

- `MLP`: shared per-stock MLP over flattened `seq_len * feature_dim`.
- `LSTM`: shared per-stock LSTM over `[seq_len, feature_dim]`, last hidden state to scalar.
- `ALSTM`: LSTM with attention pooling over hidden states.
- `Transformer`: per-stock transformer encoder over `[seq_len, feature_dim]`, pooled to scalar.
- `XGBoost`: flatten each stock sample into a feature vector and train a shared regressor over all stock-sample pairs.

## 2026-05-05 Paper Packaging Findings

- The user-provided `acl_latex.tex` already included the expanded paper sections, but it still used the ACL preamble and had almost no formal `\cite{}` usages.
- The user’s earlier PDF draft includes a concrete reference list that can be reused to seed `custom.bib`.
- A practical PRCV submission package can be produced without the official template files, but local compilation still requires:
  - `llncs.cls`
  - `splncs04.bst`
- All current in-text citations in `paper_prcv_submission_2026-05-05.tex` are resolved by `custom.bib`.
