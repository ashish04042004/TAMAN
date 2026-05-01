# Negative result as novelty: why ResNet50 (V4) did not beat ResNet18 (V3)

This note is written for **project presentation / report**: you expected a deeper backbone to help; it did not on the held-out test set. That is a **valid scientific outcome** if you frame it with hypotheses and evidence.

---

## 1. What we observed (one slide)

| Run | Backbone | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |
|-----|----------|------------|---------------|-----------|
| **V3** | ResNet18 | **46.37** | **76.14** | **0.5456** |
| **V4** | ResNet50 | 48.33 | 76.29 | 0.5438 |

**One sentence:** *Under the same multimodal pipeline, data cleaning, and training budget (15 epochs), increasing backbone depth did not improve generalization; test error moved slightly in the wrong direction.*

---

## 2. How to say it aloud (30 seconds)

> “We hypothesized that a deeper CNN (ResNet50) would capture richer visual cues and improve AQI prediction. Instead, V4 was slightly worse than V3 on the test split. We treat this as a **controlled negative result**: same splits, same metadata, same loss—only the image backbone changed. We then analyze **capacity vs data**, **train–validation behaviour**, and **how much each modality is used**.”

---

## 3. Three honest “why” hypotheses (professor-friendly)

### A. Capacity vs ~15k effective samples (overfitting / optimization)

Deeper networks have **more parameters** and can fit training noise more easily if regularization and optimization are not retuned.

**Reproducible numbers** (see `outputs/analysis_resnet50_negative_result.json`):

- Total trainable parameters (multimodal stack, same head design): **ResNet50 ≈ 2.13×** ResNet18.
- **V4 training curve (tail):** from epoch 14 → 15, **training loss decreased** while **validation RMSE worsened** (classic late-epoch **overfitting / instability** signal on the validation split).

Use this as: *“We show parameter scaling and a concrete val-vs-train divergence at the end of training.”*

### B. Visual signal may not need ResNet50 depth

Traffic-scene AQI may depend on **coarse** cues (density, haze, lighting) that ResNet18 already represents; extra layers add parameters without new **generalizable** information—especially with label noise and limited diversity.

Use this as: *“Task complexity and label quality may not reward extra depth.”*

### C. Multimodal imbalance (who carries the prediction?)

Your model uses a **learned gate** between image features and metadata features. For the **saved V4 checkpoint**, running the fusion gate on the **validation** set showed **near-total weight on the image branch** (metadata gate ≈ 0). That is empirical evidence that, for this run, the system behaved like an **image-dominated** predictor in the gated path—worth discussing whether the deeper CNN **changed optimization dynamics** in the fusion block (not just “more CNN capacity”).

**Important honesty:** this is **one checkpoint** (best validation RMSE during training) and **one split**; do not over-claim causality. Do claim: *“We measured gate behaviour and it motivates future work (e.g., regularizing gates, freezing backbone, or per-modality learning rates).”*

---

## 4. Caveat about `train_history_v3.json`

The file `outputs/train_history_v3.json` in this workspace shows **degenerate validation metrics** (repeated values across epochs). **Do not** use it for curve plots until you re-export a clean history from the actual V3 training run.

Your **V3 test metrics file** (`outputs/eval/v3/test_metrics.json`) is still the correct headline result for V3; the negative-result story should lean on:

- V3 vs V4 **test JSON** comparison, and  
- V4 **healthy** `train_history_v4_resnet50.json` + parameter ratio + gate statistics.

---

## 5. What you implemented in the repo (reproducible)

| Artifact | Purpose |
|----------|---------|
| `src/analyze_resnet50_negative_result.py` | Regenerates `outputs/analysis_resnet50_negative_result.json` with test deltas, parameter counts, V4 tail-of-training notes, fusion-gate summary on val, and sanity flags for history files. |
| `outputs/analysis_resnet50_negative_result.json` | **Numbers for slides** (copy/paste or plot in report). |

**Command (from `E:/Btech_Project`):**

```text
python src/analyze_resnet50_negative_result.py
```

---

## 6. Optional next experiments (if reviewers ask “what would you try next?”)

- **Match optimization to capacity:** lower LR on backbone, warmup+cosine, or **freeze early layers** for several epochs.  
- **Early stopping** on validation RMSE (your own V4 tail already motivates this).  
- **Gate regularization** toward balanced weights or explicit metadata residual path.  
- **Ensemble** V3 + V4 (often stabilizes predictions).

---

## 7. One-line “novelty” claim (safe wording)

> *“We present a controlled multimodal AQI pipeline and document a **negative backbone scaling result** with quantitative analysis (parameter scaling, validation–training divergence, and measured fusion-gate behaviour), motivating depth-aware training for limited urban AQ datasets.”*
