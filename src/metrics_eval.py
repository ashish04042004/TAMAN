"""Shared regression + CPCB-bucket F1 metrics for train/validation and test."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

from aqi_utils import aqi_value_to_cpcb_category


def regression_metrics_with_cpcb_f1(targets: list[float], preds: list[float]) -> dict[str, float]:
    y_true = np.asarray(targets, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    y_true_cat = np.asarray(aqi_value_to_cpcb_category(y_true), dtype=np.int64).ravel()
    y_pred_cat = np.asarray(aqi_value_to_cpcb_category(y_pred), dtype=np.int64).ravel()
    f1_macro = float(f1_score(y_true_cat, y_pred_cat, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true_cat, y_pred_cat, average="weighted", zero_division=0))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1": f1_macro,
    }
