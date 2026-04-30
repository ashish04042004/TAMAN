# Improvement Round V3 Notes

## Goal
Improve regression accuracy after observing large errors in `outputs/test_predictions.csv`.

## Changes Applied

1. Increased dataset size from 10k to 15k samples using real TRAQID data.
2. Enabled stronger metadata signal by including pollutant columns in model metadata input:
   - `pm25`
   - `pm10`
3. Increased training epochs from 10 to 15.
4. Enabled stratified splitting by AQI bins in `src/prepare_splits.py` to maintain AQI distribution across train/val/test.
5. Retained training improvements from V2:
   - Mixed precision (`autocast` + `GradScaler`)
   - Gradient clipping
   - Learning-rate scheduler (`ReduceLROnPlateau`)

## V3 Artifact Targets

- Model: `models/best_multimodal_aqi_v3.pt`
- Train history: `outputs/train_history_v3.json`
- Test metrics: `outputs/test_metrics_v3.json`
- Test predictions: `outputs/test_predictions_v3.csv`
