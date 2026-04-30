import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots for a model version.")
    parser.add_argument("--version", required=True, help="Version tag, e.g. v1 or v3")
    parser.add_argument("--pred-path", required=True, help="CSV path with target/pred columns")
    parser.add_argument("--history-path", required=True, help="JSON path with training history")
    parser.add_argument("--metrics-path", required=True, help="JSON path with test metrics")
    parser.add_argument("--plot-dir", required=True, help="Directory to save plots")
    args = parser.parse_args()

    pred_path = args.pred_path
    hist_path = args.history_path
    metrics_path = args.metrics_path
    plot_dir = args.plot_dir
    version = args.version
    os.makedirs(plot_dir, exist_ok=True)

    pred_df = pd.read_csv(pred_path)
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    target_col = next((c for c in pred_df.columns if c.startswith("target_")), None)
    pred_col = next((c for c in pred_df.columns if c.startswith("pred_")), None)
    if target_col is None or pred_col is None:
        raise ValueError("Prediction CSV must contain columns starting with target_ and pred_.")

    y_true = pred_df[target_col].to_numpy()
    y_pred = pred_df[pred_col].to_numpy()
    residuals = y_pred - y_true
    abs_err = np.abs(residuals)

    # 1) Scatter: actual vs predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.35, s=14)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Ideal")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(f"Actual vs Predicted AQI ({version.upper()})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{version}_actual_vs_predicted.png"), dpi=180)
    plt.close()

    # 2) Residual plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, residuals, alpha=0.35, s=14)
    plt.axhline(0.0, color="r", linestyle="--", linewidth=1.5)
    plt.xlabel("Actual AQI")
    plt.ylabel("Residual (Pred - Actual)")
    plt.title(f"Residuals vs Actual AQI ({version.upper()})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{version}_residuals_vs_actual.png"), dpi=180)
    plt.close()

    # 3) Absolute error distribution
    plt.figure(figsize=(7, 5))
    plt.hist(abs_err, bins=40, alpha=0.9)
    plt.xlabel("Absolute Error |Pred - Actual|")
    plt.ylabel("Count")
    plt.title(f"Absolute Error Distribution ({version.upper()})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{version}_absolute_error_hist.png"), dpi=180)
    plt.close()

    # 4) Training curves
    hist_df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="Train Loss")
    plt.plot(hist_df["epoch"], hist_df["rmse"], marker="o", label="Val RMSE")
    plt.plot(hist_df["epoch"], hist_df["mae"], marker="o", label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(f"Training and Validation Curves ({version.upper()})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{version}_training_curves.png"), dpi=180)
    plt.close()

    # 5) Predicted vs actual distributions
    plt.figure(figsize=(8, 5))
    plt.hist(y_true, bins=40, alpha=0.6, label="Actual AQI")
    plt.hist(y_pred, bins=40, alpha=0.6, label="Predicted AQI")
    plt.xlabel("AQI")
    plt.ylabel("Count")
    plt.title(f"Distribution: Actual vs Predicted AQI ({version.upper()})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{version}_actual_vs_pred_distribution.png"), dpi=180)
    plt.close()

    summary = {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "num_samples": int(len(pred_df)),
        "mean_abs_error": float(abs_err.mean()),
        "median_abs_error": float(np.median(abs_err)),
    }
    with open(os.path.join(plot_dir, f"{version}_plot_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
