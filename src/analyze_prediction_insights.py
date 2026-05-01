"""
Slice-wise error analysis for presentation / report (Novelty 8).

Expects a predictions CSV with target_aqi, pred_aqi, and optionally is_night, season_code
(TAMAN evaluate.py adds the latter two).

Usage:
    python src/analyze_prediction_insights.py --pred-csv outputs/test_predictions_taman_resnet18.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _find_pred_target_cols(df: pd.DataFrame) -> tuple[str, str]:
    pred = next((c for c in df.columns if c.startswith("pred_")), None)
    tgt = next((c for c in df.columns if c.startswith("target_")), None)
    if pred is None or tgt is None:
        raise ValueError("CSV needs columns like pred_aqi and target_aqi (pred_* / target_*).")
    return tgt, pred


def main() -> None:
    p = argparse.ArgumentParser(description="AQI prediction error slices for reports.")
    p.add_argument("--pred-csv", required=True, help="Path to predictions CSV")
    p.add_argument(
        "--output-json",
        default="outputs/prediction_insights_summary.json",
        help="Where to write aggregated metrics",
    )
    args = p.parse_args()

    df = pd.read_csv(args.pred_csv)
    tgt_col, pred_col = _find_pred_target_cols(df)
    err = (df[pred_col] - df[tgt_col]).abs()
    df = df.assign(_abs_err=err)

    summary: dict = {
        "n": int(len(df)),
        "overall_mae": float(err.mean()),
        "overall_rmse": float(np.sqrt((np.square(df[pred_col] - df[tgt_col])).mean())),
    }

    # AQI bins (CPCB-style coarse buckets for reporting)
    bins = [0, 50, 100, 200, 300, 400, 1000]
    labels = ["0-50", "51-100", "101-200", "201-300", "301-400", ">400"]
    cat = pd.cut(df[tgt_col], bins=bins, labels=labels, include_lowest=True)
    g = df.groupby(cat, observed=True)["_abs_err"].agg(["mean", "count"])
    summary["mae_by_actual_aqi_bin"] = {str(k): {"mae": float(v["mean"]), "n": int(v["count"])} for k, v in g.iterrows()}

    if "is_night" in df.columns:
        night_bin = np.where(df["is_night"] > 0.5, "night", "day")
        g2 = df.assign(_night_bin=night_bin).groupby("_night_bin")["_abs_err"].agg(["mean", "count"])
        summary["mae_by_night"] = {
            str(k): {"mae": float(v["mean"]), "n": int(v["count"])} for k, v in g2.iterrows()
        }
    if "season_code" in df.columns:
        g3 = df.groupby("season_code")["_abs_err"].agg(["mean", "count"])
        summary["mae_by_season_code"] = {
            str(int(k)) if not pd.isna(k) else "nan": {"mae": float(v["mean"]), "n": int(v["count"])}
            for k, v in g3.iterrows()
        }

    worst_idx = df["_abs_err"].nlargest(min(10, len(df))).index
    summary["worst_10_indices"] = [int(i) for i in worst_idx]

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = args.output_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(root, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
