# TAMAN — Temporal Adaptive Multimodal AQI Network

This project adds an optional architecture path: **short image sequences** → CNN (ResNet18 or ResNet50) → **LSTM** → temporal embedding; **metadata** → MLP; **scalar sigmoid gate** fuses branches → AQI regression.

## Enable in config

In `src/config.py` set:

```python
use_taman: bool = True
```

Tune (optional):

- `taman_seq_len` (default `4`): frames per window `(t-3…t)`.
- `taman_max_consecutive_frame_gap` (default `250`): max allowed numeric gap between sorted frame ids in one window (parsed from `image_name`, e.g. `front/1399.png` → `1399`).
- `taman_backbone`: `"resnet18"` or `"resnet50"`.
- `batch_size`: temporal mode uses **4×** more images per step — try **`4`** on 4–6 GB GPUs.

Outputs when TAMAN is on (defaults in `config.py`):

- Checkpoint: `models/best_taman_r18.pt`
- Train history: `outputs/train_history_taman_r18.json`
- After `evaluate.py`: `outputs/eval/taman_r18/test_metrics.json`, `outputs/eval/taman_r18/test_predictions.csv`

Evaluation figures (same style as v3/v4):

- Directory: `outputs/plots/taman_r18/`
- Regenerate:

```text
python src/plot_evaluation.py --version taman_r18 --pred-path outputs/eval/taman_r18/test_predictions.csv --history-path outputs/train_history_taman_r18.json --metrics-path outputs/eval/taman_r18/test_metrics.json --plot-dir outputs/plots/taman_r18
```

Files: `taman_r18_actual_vs_predicted.png`, `taman_r18_residuals_vs_actual.png`, `taman_r18_absolute_error_hist.png`, `taman_r18_training_curves.png`, `taman_r18_actual_vs_pred_distribution.png`, `taman_r18_plot_summary.json`.

## Train

From `E:/Btech_Project`:

```text
python src/train.py
```

If you see *zero valid windows*, increase `taman_max_consecutive_frame_gap` or verify filenames contain sortable numeric ids.

## Evaluate

Keep `use_taman=True` so the correct checkpoint path is used, then:

```text
python src/evaluate.py
```

Predictions CSV includes `is_night` and `season_code` (last frame of each window) for slice analysis.

## Insight summary (presentation tables)

```text
python src/analyze_prediction_insights.py --pred-csv outputs/eval/taman_r18/test_predictions.csv
```

Writes `outputs/prediction_insights_summary.json` (MAE by AQI bin, by day/night, by season code).

## Honest limitations

1. **Temporal order** uses **numeric id from the filename** after sorting rows — a proxy when GPS timestamps are not in the CSV. If ids jump between clips, windows are skipped via the gap rule.
2. **Train/val/test splits** are still row-level stratified from `prepare_splits.py`; neighboring windows can share frames across **different** split files only at dataset boundaries — acceptable for a student pipeline, but not “pure” video-level splits.
3. **ResNet18 vs ResNet50** comparison: set `taman_backbone` and distinct `taman_*_output_path` filenames between runs.

## Baseline (non-TAMAN) mode

Set `use_taman: bool = False` to keep the original single-image `MultiModalRegressor` training path.
