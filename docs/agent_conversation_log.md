# Agent conversation log (AeroSight-AQI)

This document summarizes work and discussions from Cursor agent session(s) used to build, evaluate, and document the multimodal and **TAMAN** AQI pipeline. It is a **human-readable digest**, not a verbatim transcript.

**Workspace:** `E:/Btech_Project`  
**Last updated:** 2026-05-01  

---

## Tabular test-set metrics (regenerated; canonical paths)

Values come from **`outputs/eval/<version>/test_metrics.json`** after `python src/run_all_evaluations.py` (or equivalent `evaluate.py` runs). Row-level test CSV has **2249** rows; **TAMAN** uses sliding windows so prediction row count can be **slightly lower**. **Lower MAE/RMSE is better; higher R² / F1 is better.**

| Version | Label | Backbone / model | Metrics file | Test MAE | Test RMSE | Test R² | F1 macro | F1 weighted |
|--------|-------|------------------|--------------|----------|------------|---------|----------|-------------|
| **V1** | Historical baseline (ckpt 5-D vs current 7-D) | ResNet18 | *skipped in `run_all_evaluations`* | 64.53* | 93.64* | 0.267* | — | — |
| **V3** | Best **single-image** multimodal here | ResNet18 | `outputs/eval/v3/test_metrics.json` | 46.37 | 76.14 | 0.546 | 0.435 | 0.591 |
| **V4** | ResNet50 multimodal (negative vs v3) | ResNet50 | `outputs/eval/v4/test_metrics.json` | 48.33 | 76.29 | 0.544 | 0.404 | 0.565 |
| **taman_r18** | Temporal windows + TAMAN | TAMAN (ResNet18) | `outputs/eval/taman_r18/test_metrics.json` | 15.61 | 27.32 | 0.942 | 0.653 | 0.834 |

\*V1 numbers from historical plot summary / first edition; not regenerated with current 7-column pipeline.

**Rounded for slides (2 d.p.)** — same as table above for v3/v4/taman.

**Related artifacts**

| Version | Predictions CSV | Train history JSON | Plots directory |
|--------|-----------------|--------------------|-----------------|
| V1 | *legacy* | `outputs/train_history.json` | `outputs/plots/v1/` |
| V3 | `outputs/eval/v3/test_predictions.csv` | `outputs/train_history_v3.json` | `outputs/plots/v3/` |
| V4 | `outputs/eval/v4/test_predictions.csv` | `outputs/train_history_v4_resnet50.json` | `outputs/plots/v4/` |
| taman_r18 | `outputs/eval/taman_r18/test_predictions.csv` | `outputs/train_history_taman_r18.json` | `outputs/plots/taman_r18/` |

**Rollup:** `outputs/eval/summary.json`  
**Index table:** `docs/evaluation_index.md`

---

## 1. Training setup: ResNet50 multimodal (V4), ~15k samples, 15 epochs

**Actions:** `src/model.py` ResNet50 path, `config.py` v4 artifact names, `prepare_splits.py`, `python src/train.py` on GPU.

**Outcome:** 15 epochs completed; test metrics in `outputs/eval/v4/`; v3 remains slightly better on MAE/RMSE/R² (controlled negative result for deeper backbone).

---

## 2–5. Progress display, tqdm, terminal clearing, training completion

Epoch summaries moved to **`tqdm.write`** in `src/train.py` so metrics do not stick to the tqdm bar. Guidance on clearing terminal view vs killing jobs; v4 run completed with **exit_code: 0**.

---

## 6. Evaluation after V4 training

Ran `evaluate.py` / structured eval → metrics under `outputs/eval/v4/`. **Conclusion:** v4 not an improvement over v3 on the same single-image task.

---

## 7. Evaluation charts for V4

`plot_evaluation.py` with `--version v4` → `outputs/plots/v4/`.

---

## 8. AGENT_CONTEXT and multi-version docs

Prior updates recorded v3/v4 defaults and plot commands. **Superseded** by current `AGENT_CONTEXT.md` (includes TAMAN, `run_all_evaluations`, F1, plot paths).

---

## 9. Ideas to improve + novelty (advisory)

Backbone tuning, augmentation, ablations (metadata without PM fields), ensembles, explainability, **temporal** modeling — **TAMAN** implemented as the temporal path.

---

## 10. TAMAN (Temporal Adaptive Multimodal AQI Network)

**Implemented:** `model_taman.py` (ResNet per frame → LSTM → metadata MLP → sigmoid gate → head), `TemporalAirQualityDataset` (sorted frame ids, `seq_len`, `max_consecutive_gap`), `train.py` / `evaluate.py` integration, `docs/taman.md`.

**Training:** Default `config.py` uses **`use_taman: True`**, 10 epochs, batch 4, TAMAN ResNet18; checkpoint `models/best_taman_r18.pt`.

**Evaluation:** `outputs/eval/taman_r18/test_metrics.json` and `test_predictions.csv` (includes `is_night`, `season_code` for slice analysis).

**Plots:** `outputs/plots/taman_r18/` via:

```text
python src/plot_evaluation.py --version taman_r18 --pred-path outputs/eval/taman_r18/test_predictions.csv --history-path outputs/train_history_taman_r18.json --metrics-path outputs/eval/taman_r18/test_metrics.json --plot-dir outputs/plots/taman_r18
```

**Presentation note:** TAMAN metrics are **not** directly comparable to v3/v4 without stating the **different input** (sequence vs single image).

---

## 11. `run_all_evaluations.py` and `evaluate.py` behaviour

- **`run_all_evaluations.py`** runs v1 (skip on schema mismatch), v3, v4, **taman_r18**; rewrites `docs/evaluation_index.md` and `outputs/eval/summary.json`.
- **`evaluate.py`:** Branch chosen from checkpoint **`model_type`** (`taman` vs multimodal); multimodal checkpoints can be evaluated even when `use_taman=True` in config (avoids false mismatch errors).

---

## 12. Beginner handbook and faculty study list

Added **`docs/beginner_project_handbook.md`** (workflow, architecture, glossary) and **`docs/study_topics_faculty_presentation.md`** (topic checklist before viva).

---

## 13. File / command reference (quick)

| Step | Command (from `E:/Btech_Project`) |
|------|-------------------------------------|
| Splits | `python src/prepare_splits.py` |
| Train | `python src/train.py` |
| Single eval | `python src/evaluate.py` [+ `--checkpoint` / `--metrics-out` / `--pred-out` / `--backbone`] |
| All evals | `python src/run_all_evaluations.py` |
| Plots | `python src/plot_evaluation.py --version <v1\|v3\|v4\|taman_r18> ...` |

---

## Note on scope

This log merges multiple threads: multimodal v3/v4, metrics and eval layout, TAMAN training/evaluation/plots, documentation handbooks, and **`AGENT_CONTEXT.md` / `STEP_LOG.md` / this file** maintenance. For verbatim chat, use Cursor’s own history export.
