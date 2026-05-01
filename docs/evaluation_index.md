# Evaluation runs (test split)

Canonical metrics and predictions live under `outputs/eval/<version>/`.
Regenerate with: `python src/run_all_evaluations.py`

Metrics include **MAE**, **RMSE**, **R²** (regression) and **F1** (macro over CPCB six-class buckets from continuous AQI).

| Version | Backbone | MAE | RMSE | R² | F1 (macro) | F1 (weighted) |
|---------|----------|-----|------|-----|------------|---------------|
| v3 | resnet18 | 46.3725 | 76.1407 | 0.5456 | 0.4346 | 0.5905 |
| v4 | resnet50 | 48.3335 | 76.2904 | 0.5438 | 0.4040 | 0.5646 |
| taman_r18 | taman (resnet18) | 15.6138 | 27.3191 | 0.9415 | 0.6530 | 0.8342 |

## Artifact paths

- **v3:** `outputs/eval/v3/test_metrics.json`, `outputs/eval/v3/test_predictions.csv`
- **v4:** `outputs/eval/v4/test_metrics.json`, `outputs/eval/v4/test_predictions.csv`
- **taman_r18:** `outputs/eval/taman_r18/test_metrics.json`, `outputs/eval/taman_r18/test_predictions.csv`

## Skipped / incompatible

- **v1:** Checkpoint expects metadata_dim=5 but config has 7 columns ('humidity', 'temperature', 'hour', 'is_night', 'season_code', 'pm25', 'pm10'). Retrain the checkpoint with the current schema or adjust config.

## Rollup

Machine-readable: `outputs/eval/summary.json`
