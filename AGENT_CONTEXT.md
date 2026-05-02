# Agent Context Handoff (AeroSight-AQI)

This file captures the current project context so another agent can continue work without losing state.

## Project Identity

- Project name: **AeroSight-AQI: Deep Multimodal Urban Air Quality Estimation**
- Workspace: `E:/Btech_Project`
- GitHub repo: `https://github.com/ashish04042004/AeroSight-AQI.git`
- Current branch: `main`

## Goal

Multimodal AQI **regression** using:

- **Image** input (traffic / outdoor scenes; TRAQID-style paths under `data/raw/`)
- **Metadata** (sensor / context fields)
- Continuous target: **`aqi`**

Two training paths exist (selected in `src/config.py` via **`use_taman`**; evaluation picks architecture from **checkpoint** `model_type`):

1. **Single-image multimodal** — `MultiModalRegressor` in `src/model.py` (ResNet18 or ResNet50 + gated fusion).
2. **TAMAN (temporal)** — `TAMAN` in `src/model_taman.py` (short image sequence → ResNet per frame → LSTM + metadata MLP + sigmoid gate → regression).

## Data Source and Authenticity

- Source zip: `E:/TRAQID/traqid.zip`
- Extraction script: `src/extract_traqid_subset.py`
- No synthetic rows are generated.
- Dataset built from authentic extracted records only.

## Current Dataset Schema

Stored in `data/raw/air_quality_metadata.csv`:

- `image_name`
- `humidity`, `temperature`, `hour`, `is_night`, `season_code`, `pm25`, `pm10`, `aqi`

Current split sizes (after cleaning in `prepare_splits.py`; **row-level** counts):

- Train: **10493**
- Validation: **2248**
- Test: **2249** rows in `test.csv`

**Note:** **TAMAN** builds sliding **windows** over test rows sorted by frame id; the number of **evaluation samples** can be slightly lower than raw test rows (see `outputs/eval/taman_r18/test_predictions.csv`).

## Data Processing Notes

- Split script: `src/prepare_splits.py` (AQI-bin stratification, numeric coercion, physical-range filters).
- AQI from PM2.5/PM10 when needed: `src/aqi_utils.py` (EPA-style breakpoints).
- Metadata z-score mean/std computed on **train** only (`dataset.compute_stats`).

## Model + Training Pipeline

| File | Role |
|------|------|
| `src/config.py` | All paths, hyperparameters, `use_taman`, TAMAN knobs, multimodal backbone. |
| `src/model.py` | `MultiModalRegressor` (single image). |
| `src/model_taman.py` | `TAMAN` + shared `_build_resnet_backbone`. |
| `src/dataset.py` | `AirQualityDataset` (one image); `TemporalAirQualityDataset` (sequence). |
| `src/train.py` | Training loop, weighted Huber loss, AdamW, ReduceLROnPlateau, checkpoint on best **val RMSE**. |
| `src/evaluate.py` | Loads checkpoint; branch from **`model_type`** in file (`taman` vs multimodal); writes metrics + CSV. |
| `src/metrics_eval.py` | MAE, RMSE, R² + CPCB **F1** (macro / weighted). |
| `src/run_all_evaluations.py` | Evaluates **v1, v3, v4, taman_r18** into `outputs/eval/<tag>/`; refreshes `docs/evaluation_index.md` and `outputs/eval/summary.json`. |

### Default `config.py` (as in repo; verify before runs)

- **`use_taman: True`** → default **`python src/train.py`** trains **TAMAN** (not single-image multimodal).
- **TAMAN:** `taman_backbone: resnet18`, `taman_seq_len: 4`, `epochs: 10`, `batch_size: 4`, `learning_rate: 5e-5`, `use_amp: False`.
- TAMAN artifacts: `models/best_taman_r18.pt`, `outputs/train_history_taman_r18.json`, `outputs/eval/taman_r18/test_metrics.json`, `outputs/eval/taman_r18/test_predictions.csv`.
- **Multimodal (when `use_taman: False`):** `multimodal_backbone: resnet50`, paths point at **v4** checkpoint `models/best_multimodal_aqi_v4_resnet50.pt` and `outputs/eval/v4/…` / `outputs/train_history_v4_resnet50.json`.

### Earlier / alternate runs

- **V1:** Historical baseline; checkpoint **5-D** metadata vs current **7-D** → skipped by `run_all_evaluations.py` until retrained.
- **V3:** Single-image multimodal, **15** epochs in saved history (`outputs/train_history_v3.json`); best **single-frame** test MAE among v3 vs v4.
- **V4:** ResNet50 multimodal, **15** epochs (`outputs/train_history_v4_resnet50.json`); slightly worse than v3 on test (documented negative result).

## NaN / Instability (historical)

- AMP + dirty metadata caused non-finite validation outputs on an older run.
- **Fixes:** `use_amp = False`, LR `5e-5`, finite checks in `train.py`, cleaning in `prepare_splits.py`, gradient clipping.

## Versioned test-set metrics (regenerated)

Canonical table: **`docs/evaluation_index.md`**. Machine-readable rollup: **`outputs/eval/summary.json`**.

Metrics include **MAE**, **RMSE**, **R²**, **F1_macro**, **F1_weighted** (CPCB six-class buckets via `metrics_eval.py`). **Lower** MAE/RMSE is better; **higher** R² / F1 is better.

| Version | Backbone / model | Test MAE | Test RMSE | Test R² | F1 macro | Metrics JSON |
|---------|------------------|----------|------------|---------|----------|----------------|
| **V1** | ResNet18 (historical) | ~64.53 | ~93.64 | ~0.267 | — | *not regenerated; ckpt schema mismatch* |
| **V3** | resnet18 multimodal | **46.37** | **76.14** | **0.546** | ~0.43 | `outputs/eval/v3/test_metrics.json` |
| **V4** | resnet50 multimodal | 48.33 | 76.29 | 0.544 | ~0.40 | `outputs/eval/v4/test_metrics.json` |
| **taman_r18** | TAMAN + ResNet18 | **15.61** | **27.32** | **0.942** | ~0.65 | `outputs/eval/taman_r18/test_metrics.json` |

**Takeaways**

- **v3 vs v4 (same task):** v3 wins slightly — controlled **negative result** for “deeper ResNet50 alone” (`docs/negative_result_resnet50_analysis.md`).
- **TAMAN vs v3/v4:** Different inputs (**temporal windows** + architecture). Higher scores are **not** a fair apples-to-apples claim against single-frame multimodal without careful wording; cite as **temporal multimodal** result.

## Artifact paths per version

**V1** — Historical: `outputs/plots/v1/`, `outputs/train_history.json`; eval skipped until compatible ckpt.

**V3** — `outputs/eval/v3/`, `outputs/train_history_v3.json`, `outputs/plots/v3/`.

**V4** — `outputs/eval/v4/`, `outputs/train_history_v4_resnet50.json`, `outputs/plots/v4/`.

**TAMAN-R18** — `models/best_taman_r18.pt`, `outputs/train_history_taman_r18.json`, `outputs/eval/taman_r18/`, **`outputs/plots/taman_r18/`** (evaluation plots + `taman_r18_plot_summary.json`).

**First edition (optional)** — `editions/first_edition/` if archived.

## Plotting and evaluation commands

- **All eval JSON/CSV:** `python src/run_all_evaluations.py` (v1 skip if mismatch; v3, v4, taman_r18).
- **Single checkpoint:** `python src/evaluate.py --checkpoint … --metrics-out … --pred-out … [--backbone resnet18|resnet50]` (backbone only for multimodal if missing in ckpt).
- **Plots:** `python src/plot_evaluation.py --version <tag> --pred-path … --history-path … --metrics-path … --plot-dir …`

**TAMAN-R18 plot example:**

```text
python src/plot_evaluation.py --version taman_r18 --pred-path outputs/eval/taman_r18/test_predictions.csv --history-path outputs/train_history_taman_r18.json --metrics-path outputs/eval/taman_r18/test_metrics.json --plot-dir outputs/plots/taman_r18
```

(Documentation: `docs/taman.md`, `docs/beginner_project_handbook.md`, `docs/study_topics_faculty_presentation.md`.)

## Repo / Git State

- Remote: `https://github.com/ashish04042004/AeroSight-AQI.git`
- `.gitignore` commonly excludes: `data/raw/`, `data/processed/`, `models/`, `outputs/`, `editions/`

## What Another Agent Should Do Next

1. **Train:** `python src/train.py` (respect `use_taman` and paths in `config.py`).
2. **Evaluate all checkpoints:** `python src/run_all_evaluations.py`.
3. **Plots:** Run `plot_evaluation.py` per version (v3, v4, taman_r18; v1 only if legacy CSVs exist).
4. **Improvements:** Early stopping; ablations without `pm25`/`pm10` in metadata; `taman_backbone: resnet50` with distinct output filenames; video-level splits (research-grade).

## Important Constraint from User

- **Do not fabricate data.** Authentic records only.
