# TAMAN — Temporal Adaptive Multimodal AQI Network

**Vision-based Air Quality Index (AQI) estimation** from urban traffic scenes, combining short image sequences with environmental metadata.

> Resume / portfolio project · PyTorch · Multimodal deep learning · TRAQID dataset

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Highlights

| Metric (test set) | Best single-image (v3) | **TAMAN-R18** |
|-------------------|------------------------|---------------|
| MAE ↓ | 46.37 | **15.61** |
| RMSE ↓ | 76.14 | **27.32** |
| R² ↑ | 0.55 | **0.94** |
| F1-weighted ↑ | 0.591 | **0.834** |

- **~66% lower MAE** vs best single-image multimodal baseline (ResNet18 + metadata)
- **~50% lower MAE** vs strongest prior TRAQID baseline (AQC-Net, MAE 31.67)
- Controlled **negative result**: ResNet50 (v4) did **not** beat ResNet18 (v3) under the same pipeline

---

## What is TAMAN?

**TAMAN** (Temporal Adaptive Multimodal AQI Network) treats AQI prediction as a **temporal multimodal** problem:

1. **CNN (ResNet18)** — spatial features from each frame in a short window  
2. **LSTM** — temporal dependencies across consecutive frames  
3. **MLP** — weather / context metadata encoding  
4. **Sigmoid adaptive gate** — learns how much to trust vision vs metadata per sample  
5. **Regression head** — continuous AQI (+ CPCB six-class F1 for evaluation)

```text
[Image sequence t−3…t] → ResNet (shared) → LSTM → temporal embedding ─┐
                                                                       ├→ sigmoid gate → fused → AQI
[Metadata at t]         → Weather MLP      → weather embedding      ─┘
```

---

## Repository structure

```text
TAMAN/
├── src/                      # Training, models, evaluation, data prep
│   ├── model_taman.py        # TAMAN architecture
│   ├── model.py              # Single-image multimodal baseline
│   ├── dataset.py            # Image + temporal window datasets
│   ├── train.py / evaluate.py
│   └── create_taman_workbook.py
├── docs/                     # Design notes, evaluation index, study guide
├── models/                   # Saved checkpoints (*.pt)
├── outputs/                  # Metrics, plots, experiment workbook
│   └── reports/TAMAN-Experiment-Workbook.xlsx
├── data/                     # Local only (gitignored raw/processed)
├── requirements.txt
└── README.md
```

---

## Results snapshot

### Baselines vs TAMAN (test)

| Model | MAE | RMSE | R² | F1-macro | F1-weighted |
|-------|-----|------|-----|----------|-------------|
| v1 (historical) | 64.53 | 93.64 | 0.27 | — | — |
| v3 ResNet18 multimodal | 46.37 | 76.14 | 0.55 | 0.435 | 0.591 |
| v4 ResNet50 multimodal | 48.33 | 76.29 | 0.54 | 0.404 | 0.565 |
| **TAMAN-R18** | **15.61** | **27.32** | **0.94** | **0.653** | **0.834** |

### Ablation (architecture components)

| Configuration | MAE | RMSE | R² |
|---------------|-----|------|-----|
| CNN only | 52.84 | 81.32 | 0.49 |
| CNN + metadata | 46.37 | 76.14 | 0.55 |
| CNN + LSTM | 28.72 | 49.65 | 0.81 |
| CNN + metadata + LSTM (no adaptive fusion) | 20.43 | 35.18 | 0.90 |
| **TAMAN (full)** | **15.61** | **27.32** | **0.94** |

Canonical metrics live under `outputs/eval/` and `docs/evaluation_index.md`. Experiment workbook: `outputs/reports/TAMAN-Experiment-Workbook.xlsx`.

---

## Quick start

```bash
git clone https://github.com/ashish04042004/TAMAN.git
cd TAMAN
pip install -r requirements.txt
```

### Data (TRAQID)

1. Obtain the [TRAQID](https://doi.org/10.1145/3702250.3702260) dataset (ICVGIP 2024).  
2. Build metadata + splits (example path used in this project):

```bash
python src/extract_traqid_subset.py   # or prepare_traqid_dataset.py
python src/prepare_splits.py
```

Raw data stays under `data/raw/` / `data/processed/` (not committed).

### Train / evaluate

```bash
# Default config trains TAMAN (see src/config.py: use_taman=True)
python src/train.py

# Single-run eval (paths from config)
python src/evaluate.py

# All versioned evals → outputs/eval/<tag>/
python src/run_all_evaluations.py

# Plots
python src/plot_evaluation.py --version taman_r18 \
  --pred-path outputs/eval/taman_r18/test_predictions.csv \
  --history-path outputs/train_history_taman_r18.json \
  --metrics-path outputs/eval/taman_r18/test_metrics.json \
  --plot-dir outputs/plots/taman_r18
```

### Key config knobs (`src/config.py`)

| Setting | Typical value |
|---------|----------------|
| `use_taman` | `True` for TAMAN, `False` for single-image multimodal |
| `taman_seq_len` | `4` |
| `taman_backbone` | `resnet18` |
| `batch_size` | `4` (temporal), `8` (single-image) |
| `epochs` | `10` (TAMAN runs in this repo) |
| `learning_rate` | `5e-5` |
| Metadata | humidity, temperature, hour, is_night, season_code, pm25, pm10 |

---

## Tech stack

- **Python**, **PyTorch**, **Torchvision** (ResNet18/50)  
- **LSTM** temporal encoder, **MLP** metadata branch, **gated fusion**  
- **Pandas / NumPy / scikit-learn** — splits & metrics  
- **Huber loss** (high-AQI weighting), **AdamW**, **ReduceLROnPlateau**  
- Evaluation: MAE, RMSE, R², CPCB-bucket F1  

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [`docs/taman.md`](docs/taman.md) | TAMAN usage & limitations |
| [`docs/evaluation_index.md`](docs/evaluation_index.md) | Metrics table |
| [`docs/beginner_project_handbook.md`](docs/beginner_project_handbook.md) | Full workflow + glossary |
| [`docs/study_topics_faculty_presentation.md`](docs/study_topics_faculty_presentation.md) | Viva / presentation checklist |
| [`docs/negative_result_resnet50_analysis.md`](docs/negative_result_resnet50_analysis.md) | Why ResNet50 did not win |

---

## Dataset & citation

This work builds on **TRAQID** (Traffic-Related Air Quality Image Dataset):

> Kathalkar et al., *TRAQID - Traffic-Related Air Quality Image Dataset*, ICVGIP 2024.  
> https://doi.org/10.1145/3702250.3702260

Prior methods compared in the report (Mondal, Kalajdjieski, Nilesh, AQC-Net / Zhang) are cited in the project report and experiment workbook.

---

## Author

**Ashish Singh** · IIT (ISM) Dhanbad  
GitHub: [ashish04042004](https://github.com/ashish04042004)

---

## License

Code in this repository is provided for academic and portfolio use. Dataset rights remain with the TRAQID authors; obtain and cite TRAQID separately.
