# Task Plan

## Goal
Add conventional baseline models (`MLP`, `LSTM`, `ALSTM`, `Transformer`, `XGBoost`) to the HS300 rolling-window evaluation pipeline so they can be compared fairly with existing `TimeCMA` variants under the same data splits and metrics.

## Phases

| Phase | Status | Notes |
|---|---|---|
| 1. Inspect current training and rolling pipeline | completed | `train.py` only supports `TimeCMA`; rolling runner currently assumes embedding generation for all methods. |
| 2. Design baseline integration strategy | completed | Same rolling windows and metrics; skip embeddings for conventional baselines; use a separate baseline training entry. |
| 3. Implement baseline model code and trainer | completed | Added `MLP`, `LSTM`, `ALSTM`, `Transformer`, `XGBoost`. |
| 4. Extend rolling runner and scripts | completed | Added baseline method names, conditional embedding logic, and remote script. |
| 5. Verify syntax and document usage | completed | Compile, shell syntax, diff check, README update all passed. |

## Key Decisions
- Baselines will use the same market data, rolling splits, target definition, and evaluation metrics as `TimeCMA`.
- Conventional baselines will not require prompt embeddings.
- Graph-free baselines will be trained from market sequences only.
- `XGBoost` will be implemented as a per-stock shared regressor over flattened sequence features, then reshaped back to `[num_samples, num_nodes]` for evaluation.

## Risks
- `xgboost` may not be installed on the remote environment.
- Existing runner currently couples embedding generation to all methods; must avoid wasting time or breaking existing TimeCMA runs.
- Need to preserve current ablation methods and outputs.

## 2026-05-05 Paper Packaging

| Phase | Status | Notes |
|---|---|---|
| 1. Inspect current ACL manuscript and generated drafts | completed | Confirmed existing `acl_latex.tex` is ACL-style, lacks wired citations, and depends on missing `custom.bib`. |
| 2. Convert manuscript to PRCV/LNCS-ready structure | completed | Added `paper_prcv_submission_2026-05-05.tex` with LNCS preamble and integrated paper content. |
| 3. Build bibliography and connect citations | completed | Added `custom.bib` and inserted formal citations across introduction, related work, and method. |
| 4. Verify citation coverage and document remaining dependencies | completed | All `\\cite{}` keys resolve in the new bib; official `llncs.cls` and `splncs04.bst` are still required from the PRCV package. |
