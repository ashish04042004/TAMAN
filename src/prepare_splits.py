import os

import pandas as pd
from sklearn.model_selection import train_test_split

from aqi_utils import compute_aqi_from_pollutants
from config import Config


def main():
    cfg = Config()
    input_csv = "data/raw/air_quality_metadata.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    required = ["image_name"]
    pollutant_cols = ["pm25", "pm10"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not any(col in df.columns for col in pollutant_cols):
        raise ValueError("Input CSV must contain at least one pollutant column: pm25 or pm10.")

    if "aqi" not in df.columns:
        df["aqi"] = df.apply(
            lambda row: compute_aqi_from_pollutants(
                pm25=row["pm25"] if "pm25" in df.columns else None,
                pm10=row["pm10"] if "pm10" in df.columns else None,
            ),
            axis=1,
        )
    # Strict numeric coercion for regression-critical columns.
    numeric_cols = ["aqi", "humidity", "temperature", "hour", "pm25", "pm10"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["aqi"])

    # Remove physically implausible sensor rows to avoid unstable training.
    if "humidity" in df.columns:
        df = df[(df["humidity"] >= 0.0) & (df["humidity"] <= 100.0)]
    if "temperature" in df.columns:
        df = df[(df["temperature"] >= -10.0) & (df["temperature"] <= 55.0)]
    if "pm25" in df.columns:
        df = df[(df["pm25"] >= 0.0) & (df["pm25"] <= 1000.0)]
    if "pm10" in df.columns:
        df = df[(df["pm10"] >= 0.0) & (df["pm10"] <= 1000.0)]
    df = df[(df["aqi"] >= 0.0) & (df["aqi"] <= 1000.0)]
    for col in cfg.metadata_columns:
        if col not in df.columns:
            df[col] = 0.0
    metadata_cols = list(cfg.metadata_columns)
    df[metadata_cols] = df[metadata_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=metadata_cols + ["aqi"])
    df["aqi_bin"] = pd.qcut(df["aqi"], q=10, labels=False, duplicates="drop")

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["aqi_bin"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["aqi_bin"]
    )
    train_df = train_df.drop(columns=["aqi_bin"])
    val_df = val_df.drop(columns=["aqi_bin"])
    test_df = test_df.drop(columns=["aqi_bin"])

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print("Saved train/val/test CSV files in data/processed")


if __name__ == "__main__":
    main()
