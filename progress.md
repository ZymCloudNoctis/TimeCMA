# Progress Log

## 2026-05-01

- Inspected `train.py`, `TimeCMA.py`, `data_loader_multistock.py`, and `metrics.py`.
- Confirmed current pipeline only supports `TimeCMA`.
- Confirmed `MultiStockSaveDataset` can support non-embedding baselines.
- Chosen implementation approach:
  - add a separate baseline trainer
  - extend rolling runner with baseline method names
  - generate embeddings only for methods that need them
- Added `models/baselines.py` with:
  - `MLPRegressor`
  - `LSTMRegressor`
  - `ALSTMRegressor`
  - `TransformerRegressor`
- Added `train_baselines.py`:
  - torch training for `mlp/lstm/alstm/transformer`
  - `xgboost` training over flattened stock-sample pairs
  - same output artifacts and metrics format as existing `train.py`
- Extended `run_rolling_experiments.py`:
  - method families: `timecma` vs `baseline`
  - baseline methods skip embedding generation
  - baseline methods call `train_baselines.py`
- Added remote runner `scripts/Rolling_HS300_Baselines.sh`
- Updated `README.md` with baseline usage notes
- Verification passed:
  - Python compile check
  - `bash -n`
  - `git diff --check`

## 2026-05-05

- Inspected `/Users/yiming/Downloads/acl_latex.tex` and confirmed it already contains most of the integrated paper text but still uses ACL style and does not have wired bibliography citations.
- Reused the original PDF draft reference list as the citation source of truth.
- Added `paper_prcv_submission_2026-05-05.tex`:
  - LNCS / PRCV-oriented main file
  - integrated manuscript body
  - main results and ablation tables inline
  - `splncs04` bibliography style
- Added `custom.bib` with all citation entries used by the paper.
- Verified that every `\cite{}` key in the new PRCV TeX file exists in `custom.bib`.
- Added `paper_prcv_compile_notes_2026-05-05.md` documenting the remaining need for official PRCV template files (`llncs.cls`, `splncs04.bst`) and the compile sequence.
