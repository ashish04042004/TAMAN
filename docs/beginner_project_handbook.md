# AeroSight-AQI: Beginner’s Handbook (Workflow, Architecture, Tools, and Terms)

This document explains **what the project does**, **how the code is organized**, **which technologies are used and why**, and **what common ML terms mean** in *this* repository. It is written for someone new to hands-on deep learning but preparing to discuss the work with professors.

---

## 1. What problem does this project solve?

**Goal:** Predict a number called **AQI (Air Quality Index)** for a location/time using:

1. **Images** of outdoor scenes (traffic / sky / city views from the **TRAQID** dataset style).
2. **Metadata** (numbers describing weather, time of day, season, and pollutant readings).

In machine learning language, this is **supervised regression**: the model learns from many examples where both inputs and the correct AQI are known, then tries to predict AQI for new examples.

**Why multimodal?** “Multimodal” means **more than one type of input**. Here: **vision** (image) + **tabular** (metadata). The hypothesis is that **what the scene looks like** and **measured conditions** together carry more information than either alone.

---

## 2. Big-picture workflow (end to end)

Think of the pipeline as a factory line:

```text
Raw data  →  Clean CSV  →  Train/val/test splits  →  Training  →  Saved model  →  Test evaluation  →  Plots & reports
```

| Step | What happens | Main files |
|------|----------------|-------------|
| 1 | Get images + labels (TRAQID zip or download) | `src/extract_traqid_subset.py`, `README.md` |
| 2 | Build one metadata CSV listing each image path and columns | `extract_traqid_subset.py` → `data/raw/air_quality_metadata.csv` |
| 3 | Optional: different TRAQID folder layout | `src/prepare_traqid_dataset.py` |
| 4 | Split rows into train / validation / test; compute AQI if needed; filter bad rows | `src/prepare_splits.py`, `src/aqi_utils.py` |
| 5 | Train a neural network; save **checkpoint** with best validation | `src/train.py`, `src/config.py` |
| 6 | Run model on **test** split; save metrics + prediction CSV | `src/evaluate.py`, `src/run_all_evaluations.py` |
| 7 | Optional charts | `src/plot_evaluation.py` |

**Important:** Large folders (`data/raw`, `data/processed`, `models`, `outputs`) may be gitignored on your machine; the **code** still describes how to reproduce everything.

---

## 3. Repository map (what each part is for)

| Path | Role |
|------|------|
| `src/config.py` | Single place for paths, hyperparameters, column names, TAMAN on/off. |
| `src/dataset.py` | Loads one image per row **or** a **sequence** of images (TAMAN); applies image transforms; **standardizes** metadata. |
| `src/model.py` | **MultiModalRegressor**: one image + metadata → AQI. |
| `src/model_taman.py` | **TAMAN**: short image sequence + metadata → AQI; shared ResNet builder. |
| `src/train.py` | Training loop, loss, optimizer, scheduler, checkpoint saving. |
| `src/evaluate.py` | Loads a checkpoint, runs test set, writes JSON + CSV. |
| `src/metrics_eval.py` | MAE, RMSE, R² + **F1** after bucketing AQI into CPCB categories. |
| `src/aqi_utils.py` | AQI from PM2.5/PM10 (EPA-style breakpoints); CPCB **class** labels for F1. |
| `src/prepare_splits.py` | Stratified split, cleaning, optional AQI computation. |
| `src/run_all_evaluations.py` | Evaluates v1/v3/v4/taman checkpoints into `outputs/eval/<tag>/`. |
| `src/plot_evaluation.py` | Matplotlib figures from predictions + history. |
| `src/analyze_prediction_insights.py` | Slice summaries (e.g. by AQI bin). |
| `src/analyze_resnet50_negative_result.py` | Compares v3 vs v4 for the “deeper CNN didn’t help” story. |
| `docs/` | Human-readable notes (evaluation index, TAMAN, negative result, this handbook). |
| `STEP_LOG.md` | Chronological lab notebook of what was implemented when. |
| `requirements.txt` | Python libraries to install. |

---

## 4. Technologies used (and why)

| Technology | Plain meaning | Why it appears here |
|------------|----------------|---------------------|
| **Python** | Programming language | Glue for data, training, evaluation. |
| **PyTorch** | Deep learning framework | Defines neural networks (`nn.Module`), runs math on GPU, autograd for training. |
| **Torchvision** | Vision utilities | **ResNet** image models, image **transforms** (resize, normalize). |
| **Pandas** | Table data tool | Read/write CSV, clean columns. |
| **NumPy** | Fast numeric arrays | Metrics, array math. |
| **scikit-learn** | Classical ML helpers | `train_test_split`, stratification, `f1_score`, `r2_score`, etc. |
| **Pillow (PIL)** | Image loading | Open JPG/PNG from disk. |
| **tqdm** | Progress bars | See epoch progress during training. |
| **Matplotlib** | Plotting | Evaluation figures. |

**GPU (CUDA):** Training convolutional networks on many images is much faster on a **GPU**. The code uses `cuda` when available (`train.py`, `evaluate.py`).

---

## 5. Data: what each column means (current TRAQID-style pipeline)

After `extract_traqid_subset.py`, typical columns include:

| Column | Meaning |
|--------|---------|
| `image_name` | Relative path from `data/raw/` to the image file. |
| `humidity`, `temperature` | Weather-related scalars. |
| `hour` | Hour of day (0–23) as a number. |
| `is_night` | 1 if night, 0 if day (derived from dataset fields). |
| `season_code` | Encoded season (e.g. 0/1/2). |
| `pm25`, `pm10` | Pollutant concentrations used for physics-based AQI and as model inputs. |
| `aqi` | Target value to predict (regression label). |

**Standardization (z-score):** For each metadata column, the code subtracts the **mean** and divides by **std** computed on the **training** set only, so all features have comparable scale. Terms: **mean**, **standard deviation**, **z-score**.

**Train / validation / test split:** Data is divided so the model is trained on one portion, tuned/monitored on **validation**, and finally scored on **test** which should not influence training. Here, splits are **stratified** by binned AQI (`aqi_bin`) so each split has similar distribution of pollution levels.

---

## 6. Architecture A — `MultiModalRegressor` (single image)

**File:** `src/model.py`

**Inputs:**

- Image tensor: shape roughly `(batch, 3, 224, 224)` — **3** color channels (RGB), **224×224** pixels after transforms.
- Metadata vector: length = number of metadata columns (currently **7**).

**Blocks (in order):**

1. **Backbone (ResNet18 or ResNet50)**  
   - A **Convolutional Neural Network (CNN)** pretrained on ImageNet (via `torchvision` weights).  
   - The final classification layer is removed (`fc = Identity`); the network outputs a **feature vector** instead of 1000 classes.  
   - **ResNet** = residual network; **ResNet50** is deeper (more layers) than **ResNet18**.

2. **Metadata encoder**  
   - Small **fully connected (Linear)** layers with **ReLU**, **BatchNorm**, **Dropout**.  
   - **ReLU:** activation that introduces nonlinearity.  
   - **BatchNorm:** stabilizes training by normalizing activations within a batch.  
   - **Dropout:** randomly zeros neurons during training to reduce **overfitting**.

3. **Fusion gate (novelty in this project)**  
   - Concatenates image features + metadata features.  
   - A small network outputs **two weights** that sum to 1 (**Softmax**): weight for image branch and weight for metadata branch.  
   - Each branch is **scaled** by its weight, then concatenated again.  
   - **Interpretation:** the model can learn to trust image vs metadata **differently per sample**.

4. **Regressor head**  
   - More Linear layers → **single output** = predicted AQI.

**Output:** One continuous number per sample (predicted AQI).

---

## 7. Architecture B — TAMAN (Temporal Adaptive Multimodal AQI Network)

**File:** `src/model_taman.py`, **dataset:** `TemporalAirQualityDataset` in `src/dataset.py`

**Idea:** Air quality can depend on **how the scene changes over a short window**, not only one frame.

**Inputs:**

- **Sequence** of `seq_len` images (default **4**): tensor shape `(batch, T, 3, H, W)`.
- Metadata from the **last** timestep in the window (same 7 columns, standardized).

**Blocks:**

1. **Same ResNet backbone** applied to each frame (shared weights).  
2. **LSTM (Long Short-Term Memory):** a **recurrent** network that reads the sequence of frame features and keeps a **hidden state** — used here to summarize temporal patterns.  
3. **Weather MLP:** metadata → dense embedding.  
4. **Scalar sigmoid gate:** one number in $(0,1)$ blends **temporal embedding** vs **weather embedding** (different style of fusion than the 2-way Softmax gate in the single-image model).  
5. **Head:** dense layers → predicted AQI.

**Temporal windows:** Rows are sorted by a **frame id** parsed from the filename. A window is valid if consecutive frame ids do not jump by more than `taman_max_consecutive_frame_gap`.

**Honest limitation:** Frame order is a **proxy** for true time if the CSV does not carry precise timestamps; see `docs/taman.md`.

---

## 8. Training strategy (what `train.py` actually does)

| Concept | In this project |
|---------|------------------|
| **Loss function** | **Weighted Huber loss**: behaves like MSE for small errors but is less sensitive to huge outliers past a threshold (`huber_delta`). **Weighted** means rows with **high AQI** (above `high_aqi_threshold`) get a higher multiplier so the model pays more attention to dangerous pollution levels. |
| **Optimizer** | **AdamW**: adaptive learning rate method with **weight decay** (L2-style regularization on weights). |
| **Learning rate** | Small value (`5e-5`): typical for fine-tuning pretrained CNNs. |
| **Scheduler** | **ReduceLROnPlateau**: if validation RMSE stops improving, learning rate is multiplied by 0.5 after `patience` epochs. |
| **Gradient clipping** | Limits gradient norm to avoid exploding updates (`clip_grad_norm_`). |
| **AMP (optional)** | Automatic Mixed Precision — faster on some GPUs but was disabled when it caused instability. |
| **Epoch** | One full pass over the training dataset. |
| **Batch** | A small group of samples processed together (`batch_size`; smaller for TAMAN because memory holds `seq_len` images per sample). |
| **Checkpoint** | Saved file (`.pt`) with `model_state_dict` and metadata mean/std for reproducible inference. |
| **Best model** | Here, checkpoint is updated when **validation RMSE** improves. |

**Validation metrics during training:** Same regression + CPCB F1 helper as test (`metrics_eval.py`).

---

## 9. Evaluation metrics (what the numbers mean)

All are computed on **held-out test** predictions unless stated otherwise.

| Metric | Name expansion | Meaning |
|--------|----------------|---------|
| **MAE** | Mean Absolute Error | Average $| \text{true} - \text{pred} |$. Same units as AQI. Lower is better. |
| **RMSE** | Root Mean Squared Error | Penalizes large errors more than MAE. Lower is better. |
| **R²** | Coefficient of determination | How much variance in AQI is explained by predictions (1 = perfect, 0 = no better than predicting the mean). Higher is better. |
| **F1 (macro)** | F1-score, macro average | After mapping continuous AQI to **discrete CPCB categories** (Good, Satisfactory, …), treat as classification: F1 balances **precision** and **recall**. **Macro** averages equally across classes (rare classes matter as much as frequent ones). |
| **F1 (weighted)** | F1-score, weighted | Same but each class weighted by support — frequent classes dominate. |

**CPCB:** Central Pollution Control Board (India) style **six bands** for AQI; implemented in `aqi_value_to_cpcb_category` in `aqi_utils.py`.

**Regression vs classification:** The model still **predicts a real number**; F1 is a **secondary** view after **bucketing** predictions and truth into classes.

---

## 10. Model versions (v1, v3, v4, TAMAN) — how to explain them simply

| Tag | Idea |
|-----|------|
| **v1** | Early checkpoint with **fewer metadata dimensions**; cannot be evaluated with today’s 7-column config without retraining. Historical baseline (~MAE 64.5) lives in old plots/notes. |
| **v3** | Single-image multimodal, **ResNet18** backbone — best **multimodal single-frame** test MAE among v3 vs v4 in your table. |
| **v4** | Same pipeline, **ResNet50** — slightly worse on test: a useful **negative result** (“deeper backbone did not help under same setup”). See `docs/negative_result_resnet50_analysis.md`. |
| **taman_r18** | **Different setup**: sequence of frames + LSTM + gate — **not directly comparable** to v3/v4 without careful wording, but numbers are in `docs/evaluation_index.md`. |

**Presentation tip:** Say clearly whether you compare **single-image multimodal** (v3 vs v4) or **temporal multimodal** (TAMAN) vs baselines.

---

## 11. Glossary of terms you will hear in viva / presentations

**Activation function:** Nonlinearity (e.g. ReLU) so the network can learn curves, not just linear blends.

**Autograd:** PyTorch’s automatic computation of **gradients** for backpropagation.

**Backbone:** The main CNN that turns pixels into features.

**BatchNorm:** Normalization layer to stabilize training.

**Bias–variance tradeoff:** Simple idea — too simple a model **underfits**; too complex **overfits** training noise.

**Checkpoint:** Saved weights + training metadata.

**CNN / ConvNet:** Network using convolution layers; strong for images.

**Dropout:** Regularization by randomly dropping units during training.

**Embedding:** A learned vector representation (here, compact numeric summary of metadata or time).

**Epoch:** One full sweep through the training set.

**Fine-tuning:** Starting from pretrained weights and continuing training on your task.

**Fusion:** Combining image and non-image information.

**Gate / gating:** A learned mechanism that controls how much each branch contributes.

**Gradient / backpropagation:** How the loss signals each weight to move to reduce error.

**Hidden state (LSTM):** Memory carried across sequence steps.

**Huber loss:** Robust loss between L1 and L2 behavior.

**Hyperparameter:** Setting chosen by you (learning rate, batch size, epochs), not learned from data.

**Inference / evaluation:** Using a trained model without updating weights.

**Linear layer (`nn.Linear`):** Matrix multiply + bias — the basic “fully connected” layer.

**LSTM:** Recurrent architecture for sequences.

**Metadata / tabular features:** Non-image columns in a table.

**Multimodal:** Multiple input modalities (image + table).

**Normalization (image):** Rescale pixel channels using ImageNet mean/std so pretrained models see familiar input scale.

**Overfitting:** Model memorizes training set; validation/test get worse.

**Pretraining:** Training (usually on a large dataset like ImageNet) before your task.

**Regression:** Predicting a continuous number (AQI), not a class label only.

**Residual:** Skip connections in ResNet that ease training of deep networks.

**Scheduler:** Rule to change learning rate during training.

**Softmax:** Turns logits into probabilities that sum to 1.

**Stratified split:** Keeps label distribution similar across splits.

**Supervised learning:** Learning from input–output pairs.

**Target / label / ground truth:** The true AQI you want the model to predict.

**Tensor:** Multi-dimensional array (on CPU or GPU).

**Train–validation–test:** Standard three-way split of data roles.

**Weight decay:** Regularization penalizing large weights.

**Zero division in F1:** Handled with `zero_division=0` in sklearn calls when a class never appears.

---

## 12. What you should say about limitations (professors appreciate honesty)

1. **Pollutants in metadata (`pm25`, `pm10`):** They are physically related to AQI; including them can make the task **easier** and inflate metrics versus “image-only” sensing. Mention **ablation** (train without them) as future work if asked.  
2. **TAMAN temporal order** depends on filename ids — a proxy, not guaranteed perfect timeline.  
3. **Splits are row-level**, not “video-level”; adjacent windows can correlate — acceptable for coursework but not ideal for a pure video benchmark.  
4. **v1 checkpoint** incompatible with current schema — show you understand **feature schema drift**.

---

## 13. Quick commands cheat sheet

```text
pip install -r requirements.txt
python src/prepare_splits.py          # after air_quality_metadata.csv exists
python src/train.py                   # uses config: TAMAN or multimodal
python src/evaluate.py                # default paths from config
python src/run_all_evaluations.py     # v3, v4, taman_r18 (+ v1 skip if mismatch)
```

---

## 14. Related documents in this repo

- `README.md` — setup and high-level overview (some columns differ from TRAQID subset path; trust `config.py` + `extract_traqid_subset.py` for your real run).  
- `docs/evaluation_index.md` — latest test metrics table.  
- `docs/taman.md` — TAMAN-specific notes.  
- `docs/negative_result_resnet50_analysis.md` — v3 vs v4 story.  
- `STEP_LOG.md` — implementation timeline (numbered steps; includes TAMAN, eval consolidation, plot commands).  
- `AGENT_CONTEXT.md` — agent handoff: current config defaults, metrics table, artifact paths, next steps.  
- `docs/agent_conversation_log.md` — digest of major decisions and commands across sessions.  
- **`docs/study_topics_faculty_presentation.md`** — what to study before presenting (companion file).

---

*This handbook describes the codebase design as of the documented evaluation layout. If you change `metadata_columns` or model paths, update `src/config.py` and rerun evaluations.*
