# Multimodal AQI Regression (Image + Weather Metadata)

This project predicts **AQI (Air Quality Index)** using:
- Outdoor sky/city image
- Weather and context metadata (`humidity`, `temperature`, `wind_speed`, `pressure`, `hour`, `latitude`, `longitude`)

It is designed for a final-year BTech project with a clear novelty:
- **Adaptive modality fusion gate** (dynamic weighting of image vs metadata per sample).

## 1) Project Structure

- `src/prepare_splits.py`: prepares train/val/test CSV files.
- `src/dataset.py`: PyTorch dataset and metadata standardization.
- `src/model.py`: multimodal model with adaptive fusion gate.
- `src/train.py`: training + validation tracking.
- `src/evaluate.py`: test metrics and prediction export.
- `STEP_LOG.md`: record of each implementation step.

## 2) Dataset Strategy (Internet-available + citable ground truth)

Recommended build path:
1. Download TRAQID public dataset (ICVGIP 2024) containing outdoor traffic images + AQI/PM2.5/PM10 + weather metadata.
2. Convert TRAQID schema into project schema using `src/prepare_traqid_dataset.py`.
3. Build train/val/test splits using `src/prepare_splits.py`.

Ground truth is AQI computed from public pollutant measurements (PM2.5/PM10) using a standard citable formula.

## 3) Required Raw Data Format

Create `data/raw/air_quality_metadata.csv` with columns:

- `image_name`
- `humidity`
- `temperature`
- `wind_speed`
- `pressure`
- `hour`
- `latitude`
- `longitude`
- `pm25` and/or `pm10` (source pollutant measurements)
- `aqi` (optional precomputed target; auto-computed if missing)

For TRAQID conversion, `image_name` stores relative paths from `data/raw/` (e.g., `traqid_download/Images/1/Front/11201.jpg`).

## 4) Setup

```bash
pip install -r requirements.txt
```

## 5) Train + Evaluate

```bash
python -m gdown --folder "https://drive.google.com/drive/folders/1qkHjzeYPTlJiyBh-xCFq_fmSh0qq9UPV" -O data/raw/traqid_download
python src/prepare_traqid_dataset.py
python src/prepare_splits.py
python src/train.py
python src/run_all_evaluations.py
```

Outputs (see also `docs/evaluation_index.md`):
- Best model: `models/best_multimodal_aqi_v4_resnet50.pt` (or versioned name in `src/config.py`)
- Training history: `outputs/train_history_*.json`
- Test metrics + predictions (structured): `outputs/eval/<version>/test_metrics.json`, `outputs/eval/<version>/test_predictions.csv`

## 6) Suggested Report Metrics

- MAE
- RMSE
- R²
- F1 (macro over CPCB six-class buckets from continuous AQI; see `src/metrics_eval.py`)

## 7) Next Upgrade (for extra novelty)

- Time-aware fusion using short timestamp history windows.
- Self-supervised image pretraining on unlabeled sky images before regression fine-tuning.

## 8) AQI Computation

AQI is computed automatically in `src/prepare_splits.py` using `src/aqi_utils.py`:
- If `aqi` column is missing, the pipeline computes AQI from `pm25` and/or `pm10`.
- Current implementation follows standard breakpoint interpolation (US EPA style) for PM2.5 and PM10.
