# Test evaluation outputs

Each subfolder holds **`test_metrics.json`** and **`test_predictions.csv`** for one checkpoint line (same test split, current metadata schema).

- **`v3/`** — `models/best_multimodal_aqi_v3.pt` (ResNet18-style multimodal)
- **`v4/`** — `models/best_multimodal_aqi_v4_resnet50.pt` (ResNet50)

**Regenerate everything:** from repo root run `python src/run_all_evaluations.py` (see `docs/evaluation_index.md`).

**Single run:** `python src/evaluate.py --checkpoint models/... --metrics-out outputs/eval/<tag>/test_metrics.json --pred-out outputs/eval/<tag>/test_predictions.csv --backbone resnet18|resnet50`
