# Agent Context Handoff (AeroSight-AQI)

This file captures the current project context so another agent can continue work without losing state.

## Project Identity

- Project name: **AeroSight-AQI: Deep Multimodal Urban Air Quality Estimation**
- Workspace: `E:/Btech_Project`
- GitHub repo: `https://github.com/ashish04042004/AeroSight-AQI.git`
- Current branch: `main`

## Goal

Multimodal AQI regression using:
- Image input (traffic scene images from TRAQID)
- Metadata input (sensor/environmental fields)
- Continuous target: `aqi`

## Data Source and Authenticity

- Source zip: `E:/TRAQID/traqid.zip`
- Extraction script: `src/extract_traqid_subset.py`
- No synthetic rows are generated.
- Dataset currently built from authentic extracted rows only.

## Current Dataset Schema

Stored in `data/raw/air_quality_metadata.csv`:
- `image_name`
- `humidity`
- `temperature`
- `hour`
- `is_night`
- `season_code`
- `pm25`
- `pm10`
- `aqi`

Current split sizes (after cleaning):
- Train: `10493`
- Validation: `2248`
- Test: `2249`

## Data Processing Notes

- Split script: `src/prepare_splits.py`
- AQI-bin stratification is used for split balance.
- Numeric coercion + invalid row removal for corrupted/implausible sensor records.
- Metadata columns are strictly numeric before training.

## Model + Training Pipeline

- Model file: `src/model.py` (`MultiModalRegressor`)
- Train file: `src/train.py`
- Eval file: `src/evaluate.py`
- Config file: `src/config.py`

Current important config (see `src/config.py`; defaults point at **v4 ResNet50** artifacts):
- Backbone: **ResNet50** (`src/model.py`)
- `epochs = 15`
- `batch_size = 8`
- `num_workers = 0`
- `learning_rate = 5e-5`
- `use_amp = False`
- Target: `aqi`
- Metadata used: `humidity, temperature, hour, is_night, season_code, pm25, pm10`
- Checkpoint / outputs: `models/best_multimodal_aqi_v4_resnet50.pt`, `outputs/eval/v4/test_metrics.json`, `outputs/train_history_v4_resnet50.json`, `outputs/eval/v4/test_predictions.csv`

Earlier runs (not necessarily current defaults):
- **V1 / initial pipeline:** ResNet18, **10** training epochs (see `outputs/train_history.json`).
- **V3:** ResNet18; saved history has **15** epochs (`outputs/train_history_v3.json`). Best test metrics below are the v3 reference.

## NaN Root Cause and Fix Status

Observed issue:
- V3 run with AMP + broader dirty metadata produced non-finite predictions in validation.

Root causes identified:
1. Numerical instability with fp16 mixed precision on this setup.
2. Corrupted/extreme metadata records (e.g., implausible temperature values) in extracted data.

Implemented fixes:
- Disabled AMP (`use_amp = False`).
- Lowered LR to `5e-5`.
- Added explicit finite checks in training/evaluation.
- Removed non-physical/corrupt rows in preprocessing.
- Kept authentic-only data (no fabricated values).

## Versioned test-set results (all versions)

Canonical **regenerated** metrics live under `outputs/eval/<version>/` (see `docs/evaluation_index.md`). Values include **F1 (macro)** on CPCB six-class buckets from continuous AQI. **Lower MAE/RMSE is better**; **higher R² / F1 is better**.

| Version | Backbone | Train epochs (saved history) | Test MAE | Test RMSE | Test R² | F1 macro | Metrics JSON |
|--------|----------|-------------------------------|-----------|------------|---------|----------|----------------|
| **V1** | ResNet18 | 10 | 64.53 (historical baseline) | 93.64 | 0.2673 | — | *checkpoint incompatible with current 7-D metadata; see `docs/evaluation_index.md`* |
| **V3** | ResNet18 | 15 | **46.37** | **76.14** | **0.5456** | ~0.43 | `outputs/eval/v3/test_metrics.json` |
| **V4** | ResNet50 | 15 | 48.33 | 76.29 | 0.5438 | ~0.40 | `outputs/eval/v4/test_metrics.json` |

**Takeaway:** On the held-out test set, **v3 remains the best** of v3 vs v4 (lowest MAE/RMSE, highest R²). v4 (ResNet50, 15 epochs, batch 8) is very close on RMSE/R² but slightly worse on MAE.

### Artifact paths per version

**V1**
- Historical metrics (pre–7-D schema): see `docs/first_edition_notes.md` / `editions/first_edition/` if archived.
- Train history: `outputs/train_history.json`
- Plots: `outputs/plots/v1/`

**V3**
- Eval: `outputs/eval/v3/test_metrics.json`, `outputs/eval/v3/test_predictions.csv`
- Train history: `outputs/train_history_v3.json`
- Plots: `outputs/plots/v3/`

**V4 (ResNet50)**
- Eval: `outputs/eval/v4/test_metrics.json`, `outputs/eval/v4/test_predictions.csv`
- Train history: `outputs/train_history_v4_resnet50.json`
- Plots: `outputs/plots/v4/`

**First edition archive (optional mirror)**

- If present under `editions/first_edition/`, metrics should match the historical v1 row above.

## Plotting and Evaluation Artifacts

Plot script:
- `src/plot_evaluation.py` (version-parameterized CLI)

Versioned plot folders:
- V1 plots: `outputs/plots/v1/`
- V3 plots: `outputs/plots/v3/`
- V4 plots: `outputs/plots/v4/`

Each folder contains:
- Actual vs predicted
- Residuals vs actual
- Absolute error histogram
- Training curves
- Actual vs predicted distribution
- Plot summary JSON

## Repo / Git State

- Repo initialized and pushed to:
  - `https://github.com/ashish04042004/AeroSight-AQI.git`
- `.gitignore` excludes large artifacts:
  - `data/raw/`, `data/processed/`, `models/`, `outputs/`, `editions/`

## What Another Agent Should Do Next

1. If training is needed again:
   - `python src/train.py`
2. Evaluate all saved checkpoints into `outputs/eval/<version>/`:
   - `python src/run_all_evaluations.py`  
   - Single checkpoint: `python src/evaluate.py --checkpoint models/... --metrics-out ... --pred-out ... --backbone resnet18|resnet50`
3. Generate versioned plots (from repo root `E:/Btech_Project`):
   - V3:
     - `python src/plot_evaluation.py --version v3 --pred-path "outputs/eval/v3/test_predictions.csv" --history-path "outputs/train_history_v3.json" --metrics-path "outputs/eval/v3/test_metrics.json" --plot-dir "outputs/plots/v3"`
   - V4:
     - `python src/plot_evaluation.py --version v4 --pred-path "outputs/eval/v4/test_predictions.csv" --history-path "outputs/train_history_v4_resnet50.json" --metrics-path "outputs/eval/v4/test_metrics.json" --plot-dir "outputs/plots/v4"`
   - V1 (historical CSVs only, if you still have legacy prediction files):
     - Point `--pred-path` / `--metrics-path` at those files; eval outputs for v1 are not auto-produced until a compatible checkpoint exists.
4. If further improvement is desired:
   - Try robust normalization/capping strategies per feature.
   - Add controlled ablations (with/without `pm25`,`pm10`) and compare.
   - Keep all outputs versioned and non-overwriting.

## Important Constraint from User

- User explicitly requested: **do not make up any data**.
- Maintain authentic extracted records only.
