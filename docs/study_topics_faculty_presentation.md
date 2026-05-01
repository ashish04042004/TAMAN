# Topics to study before presenting AeroSight-AQI to professors

Use this as a **checklist**. You do not need to be an expert in every math detail; you need to **explain your design choices**, **interpret metrics**, and **state limitations** clearly.

---

## A. Foundations (must be comfortable)

1. **Supervised learning** — input–output pairs, training vs inference.  
2. **Train / validation / test splits** — why test data must stay untouched during model decisions.  
3. **Regression vs classification** — you predict continuous AQI but also report F1 after bucketing.  
4. **Overfitting and underfitting** — symptoms; why validation matters.  
5. **Basic linear algebra intuition** — vectors, matrices, “what a layer does” at a high level.

---

## B. Neural networks and deep learning (core)

6. **Neurons, layers, nonlinearities (ReLU)** — why stacking linear layers alone is insufficient.  
7. **Loss functions** — MAE, MSE, RMSE; **Huber loss** and why it is robust to outliers.  
8. **Gradient descent** — loss → gradients → weight updates (concept only).  
9. **Backpropagation** — chain rule idea; you do not need to derive it by hand.  
10. **Optimizers** — **SGD vs Adam / AdamW**; role of **learning rate** and **weight decay**.  
11. **Batch size and epochs** — memory vs noise in gradient estimates.  
12. **Regularization** — dropout, weight decay, early stopping (concept).  
13. **Batch normalization** — one sentence: stabilizes training for deep nets.

---

## C. Computer vision (for the image branch)

14. **Convolutional neural networks (CNNs)** — filters, local patterns, hierarchy from edges to objects.  
15. **Transfer learning / pretrained models** — ImageNet pretraining; why it helps small datasets.  
16. **ResNet (ResNet18 vs ResNet50)** — depth, parameters, residual connections (high level).  
17. **Image preprocessing** — resize, crop, **normalization** with mean/std.  
18. **Data augmentation** — random crop/flip; why it is used only in training.

---

## D. Multimodal and temporal models (your novelty lines)

19. **Multimodal learning** — fusing different input types; challenges (alignment, scaling).  
20. **Attention vs gating (informal)** — your **Softmax gate** (two-way) vs **sigmoid gate** (blend) in TAMAN — be able to point to the two files (`model.py` vs `model_taman.py`).  
21. **RNN / LSTM basics** — sequence input, hidden state, why LSTM for short frame history.  
22. **Temporal windows** — what `seq_len` and `max_consecutive_gap` mean in your dataset code.

---

## E. Metrics and evaluation (they will ask numbers)

23. **MAE, RMSE, R²** — formulas intuitive; which penalizes large errors more (RMSE).  
24. **Precision, recall, F1** — for CPCB bucket classification derived from regression outputs.  
25. **Macro vs weighted F1** — class imbalance; when each is fair to report.  
26. **Stratified sampling** — why you stratify on binned AQI in `prepare_splits.py`.  
27. **Negative results** — why “ResNet50 did not beat ResNet18” is still scientific value if controlled.

---

## F. Domain: air quality (credibility)

28. **AQI definition (high level)** — index from pollutants; not identical to raw PM2.5 alone.  
29. **PM2.5 vs PM10** — particle size intuition.  
30. **EPA-style breakpoint tables** — piecewise linear sub-indices; `aqi_utils.py` implements this style.  
31. **CPCB AQI bands** — six classes used for F1 mapping.  
32. **TRAQID dataset** — what it contains; cite the paper/source your report uses.

---

## G. Engineering and reproducibility (impresses examiners)

33. **PyTorch tensors, `nn.Module`, `forward`** — how your code maps to these.  
34. **DataLoader, Dataset** — batching, workers, shuffle train not val.  
35. **Checkpoints** — what is saved (weights, mean/std); why mean/std must match training.  
36. **GPU vs CPU** — why CNN training prefers GPU.  
37. **Mixed precision (AMP)** — benefit and your project’s choice to disable when unstable.  
38. **Random seed** — reproducibility limits.

---

## H. Ethics and limitations (short but mature)

39. **Data leakage** — if metadata contains information too close to the label (e.g. pollutants vs AQI), interpret metrics carefully; ablation studies.  
40. **Generalization** — model trained on one city/dataset may not transfer.  
41. **Fairness / deployment** — health decisions need regulatory-grade monitoring (one sentence is enough for BTech unless that is your focus).

---

## I. Optional stretch (if you want depth)

42. **Huber loss derivation sketch** — kink at delta.  
43. **LSTM gating equations** — input/forget/output gates (optional).  
44. **Learning rate schedules** — ReduceLROnPlateau vs cosine schedules.  
45. **Attention mechanisms** — contrast with your simpler gates as “future work.”

---

## Presentation drill (recommended)

- **2-minute story:** problem → data → model diagram → main metric table → one limitation → conclusion.  
- **1 slide** with architecture diagram (single-image vs TAMAN).  
- **1 slide** with `docs/evaluation_index.md` numbers and a sentence on comparability.  
- Prepare answers for: *Why Huber? Why gate? Why TAMAN? Why did ResNet50 fail to beat ResNet18? What would you do next?*

---

*Pair this file with `docs/beginner_project_handbook.md` for full workflow and term explanations.*
