# First Edition Notes

## Edition
- Name: `first_edition`
- Date: 2026-04-30
- Dataset: TRAQID 10k subset extracted from `E:/TRAQID/traqid.zip`
- Task: Multimodal AQI regression (image + metadata)

## Input Features
- Image: front camera PNG images
- Metadata:
  - `humidity`
  - `temperature`
  - `hour`
  - `is_night`
  - `season_code`

## Target
- `aqi` (continuous regression target)

## Split
- Train/Val/Test generated from `data/raw/air_quality_metadata.csv`
- Script: `src/prepare_splits.py`

## Metrics (Test)
- MAE: `64.53056791178385`
- RMSE: `93.63713326304398`
- R2: `0.26734462235291834`

## Artifact Paths
- Model: `editions/first_edition/model_best_multimodal_aqi.pt`
- Metrics: `editions/first_edition/test_metrics.json`
- Predictions: `editions/first_edition/test_predictions.csv`
- Training history: `editions/first_edition/train_history.json`
