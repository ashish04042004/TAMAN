# Step Execution Log

This file records every implementation step performed for the project:
**Air Quality (AQI/PM2.5) Estimation from Sky/City Image + Weather Metadata**.

## Steps

1. Initialized an empty workspace check with `ls`.
2. Created the base project folders:
   - `src/`
   - `data/`, `data/raw/`, `data/processed/`
   - `models/`
   - `outputs/`
   - `docs/`
3. Verified local Python and pip availability.
4. Added dependency manifest in `requirements.txt`.
5. Implemented central configuration in `src/config.py`.
6. Implemented multimodal dataset loader and metadata standardization in `src/dataset.py`.
7. Implemented multimodal regression model with adaptive fusion novelty in `src/model.py`.
8. Implemented model training pipeline in `src/train.py`.
9. Implemented test evaluation and prediction export in `src/evaluate.py`.
10. Implemented dataset split utility in `src/prepare_splits.py`.
11. Added full project usage and architecture guide in `README.md`.
12. Added OpenAQ-based dataset pairing helper in `src/build_dataset_openaq.py`.
13. Added report citation and ground-truth justification notes in `docs/report_citation_notes.md`.
14. Started AQI-target conversion from PM2.5-based target.
15. Added AQI computation utility in `src/aqi_utils.py` with PM2.5/PM10 breakpoint interpolation.
16. Updated target configuration to `aqi` and model artifact path to `models/best_multimodal_aqi.pt` in `src/config.py`.
17. Updated `src/prepare_splits.py` to auto-compute `aqi` when absent, using pollutant inputs (`pm25` and/or `pm10`).
18. Updated `src/build_dataset_openaq.py` to include pollutant fields and computed `aqi` in output metadata.
19. Updated `src/evaluate.py` prediction export columns to dynamic target naming (`target_aqi`/`pred_aqi`).
20. Updated project documentation in `README.md` for AQI target and automatic AQI computation behavior.
21. Updated report citation guidance in `docs/report_citation_notes.md` to cite AQI formula source and pollutant-derived AQI labels.
22. Received user-provided zipped TRAQID dataset at `E:/TRAQID/traqid.zip`.
23. Verified zip structure and confirmed full labels file (`TRAQID.csv`) with 26,678 rows.
24. Added `src/extract_traqid_subset.py` for targeted zip extraction of aligned image-label subsets.
25. Extracted a real 10,000-sample subset from zip into `data/raw/traqid_subset/front/`.
26. Generated aligned metadata file `data/raw/air_quality_metadata.csv` with AQI + pollutant ground truth.
27. Fixed metadata fill assignment bug in `src/prepare_splits.py` for tuple/list column indexing.
28. Generated train/validation/test splits from the 10k subset into `data/processed/`.
29. Installed missing deep learning dependencies (`torch`, `torchvision`).
30. Started model training (`python src/train.py`) on the 10k real subset.
31. Investigated training crash from terminal log (`numpy.object_` metadata tensor conversion failure).
32. Fixed metadata casting in `src/dataset.py` using numeric coercion and NaN-safe float conversion before tensor creation.
33. Prepared to restart training after data-type fix.
34. Updated `src/config.py` to reduce training from 15 to 10 epochs as requested.
35. Diagnosed placeholder-heavy metadata issue (zeros in non-TRAQID fields) and confirmed concern.
36. Reworked feature schema to use real TRAQID metadata only: `humidity`, `temperature`, `hour`, `is_night`, `season_code`.
37. Updated `src/extract_traqid_subset.py` to generate `is_night` and `season_code` from TRAQID categorical fields.
38. Re-extracted aligned 10k subset metadata and regenerated data splits.
39. Restarted training with corrected metadata and 10-epoch configuration (`Epoch 1/10` running).
40. Paused CPU training and migrated environment to CUDA-enabled PyTorch (`torch 2.11.0+cu128`).
41. Verified GPU runtime availability (`torch.cuda.is_available() = True`, device: NVIDIA GeForce GTX 1650).
42. Restarted training on GPU and completed 10-epoch run successfully.
43. Ran evaluation on test split via `src/evaluate.py`.
44. Saved final test metrics in `outputs/test_metrics.json` and predictions in `outputs/test_predictions.csv`.
45. Archived baseline outputs as First Edition under `editions/first_edition/`.
46. Added First Edition documentation file `docs/first_edition_notes.md` with dataset/features/metrics/artifact paths.
47. Began V2 improvement cycle: added AMP mixed precision, gradient clipping, and ReduceLROnPlateau scheduler in `src/train.py`.
48. Updated V2 artifact paths in `src/config.py` (`best_multimodal_aqi_v2.pt`, `test_metrics_v2.json`, `train_history_v2.json`, `test_predictions_v2.csv`).
49. Updated `src/evaluate.py` to save predictions to config-driven V2 path.
50. Received improvement request due to large prediction gaps and initiated V3 improvement cycle.
51. Stopped in-progress V2 run to avoid training on outdated setup.
52. Increased data size to 15,000 real samples via `src/extract_traqid_subset.py`.
53. Updated config for V3: 15 epochs, metadata includes `pm25` and `pm10`, `num_workers=2`, and V3 artifact paths.
54. Added AQI-bin stratified splitting in `src/prepare_splits.py` and regenerated train/val/test CSVs.
55. Started V3 training (`python src/train.py`) on 15k data and 15-epoch schedule.
56. Added run documentation in `docs/improvement_round_v3_notes.md`.
57. Investigated V3 crash after epoch 1; identified NaN predictions during validation metric computation.
58. Added numerical stability guards in `src/train.py` (`torch.nan_to_num` on predictions and skip non-finite losses).
59. Reduced learning rate to `5e-5` in `src/config.py` for more stable optimization.
60. Completed corrected 12-epoch authentic-data training run and evaluated V3 outputs.
61. Added `src/plot_evaluation.py` to generate reproducible Matplotlib evaluation plots.
62. Generated V3 plots and summary under `outputs/plots/`:
    - `v3_actual_vs_predicted.png`
    - `v3_residuals_vs_actual.png`
    - `v3_absolute_error_hist.png`
    - `v3_training_curves.png`
    - `v3_actual_vs_pred_distribution.png`
    - `v3_plot_summary.json`
63. Updated `src/plot_evaluation.py` to support versioned plotting via CLI arguments.
64. Created separate version plot folders: `outputs/plots/v1/` and `outputs/plots/v3/`.
65. Generated complete V1 and V3 evaluation plot sets with isolated files per version.
66. Created `AGENT_CONTEXT.md` as a handoff context file containing complete project state, decisions, artifacts, metrics, and next actions for continuity across agents.
67. Added CPCB six-class **F1** (macro / weighted) alongside regression metrics (`src/metrics_eval.py`, `src/aqi_utils.py` with `aqi_value_to_cpcb_category`).
68. Consolidated test outputs under `outputs/eval/<version>/`; added `src/run_all_evaluations.py` and `docs/evaluation_index.md` (table + rollup `outputs/eval/summary.json`).
69. Implemented **TAMAN** temporal path: `src/model_taman.py`, `TemporalAirQualityDataset` and frame-id windowing in `src/dataset.py`, branches in `src/train.py` and `src/evaluate.py`, user docs in `docs/taman.md`.
70. Trained **TAMAN–ResNet18** on GPU (10 epochs, batch 4 per `config.py`); saved `models/best_taman_r18.pt`, `outputs/train_history_taman_r18.json`, evaluation under `outputs/eval/taman_r18/`.
71. Updated `src/run_all_evaluations.py` to include **`taman_r18`**; relaxed `src/evaluate.py` so evaluation branch follows **checkpoint `model_type`**, not a conflicting `use_taman` default when scoring multimodal checkpoints.
72. Added presentation / onboarding docs: `docs/beginner_project_handbook.md`, `docs/study_topics_faculty_presentation.md`.
73. Generated **TAMAN-R18 evaluation plots** with `src/plot_evaluation.py` into `outputs/plots/taman_r18/` (scatter, residuals, error histogram, training curves, distributions, `taman_r18_plot_summary.json`); documented command in `docs/taman.md`.
74. Refreshed **`AGENT_CONTEXT.md`**, **`STEP_LOG.md`**, and **`docs/agent_conversation_log.md`** to match the above handoff state.
