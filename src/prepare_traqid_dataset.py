from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _find_first_csv(root: Path) -> Path:
    for path in root.rglob("*.csv"):
        return path
    raise FileNotFoundError(f"No CSV file found under: {root}")


def _pick_first_existing_column(df: pd.DataFrame, choices: list[str]) -> str | None:
    for col in choices:
        if col in df.columns:
            return col
    return None


def main():
    traqid_root = Path("data/raw/traqid_download")
    images_root = traqid_root / "Images"
    out_csv = Path("data/raw/air_quality_metadata.csv")
    out_images_root = Path("data/raw/images")

    csv_path = _find_first_csv(traqid_root)
    df = pd.read_csv(csv_path)

    sequence_col = _pick_first_existing_column(df, ["Sequence", "sequence", "seq"])
    frame_col = _pick_first_existing_column(df, ["Frame", "frame", "Image", "image", "index"])
    pm25_col = _pick_first_existing_column(df, ["PM2.5", "pm25", "PM25"])
    pm10_col = _pick_first_existing_column(df, ["PM10", "pm10"])
    aqi_col = _pick_first_existing_column(df, ["AQI", "aqi"])
    temp_col = _pick_first_existing_column(df, ["Temperature", "temperature", "temp"])
    humidity_col = _pick_first_existing_column(df, ["Humidity", "humidity"])

    if sequence_col is None or frame_col is None:
        raise ValueError("Could not identify sequence/frame columns in TRAQID CSV.")
    if aqi_col is None and pm25_col is None and pm10_col is None:
        raise ValueError("Need at least AQI or pollutant columns in TRAQID CSV.")

    records = []
    for _, row in df.iterrows():
        seq = str(row[sequence_col]).strip()
        frame = str(int(row[frame_col])) if str(row[frame_col]).replace(".", "", 1).isdigit() else str(row[frame_col]).strip()
        front_path = images_root / seq / "Front" / f"{frame}.jpg"
        if not front_path.exists():
            continue

        relative_image = front_path.relative_to(out_images_root.parent)
        rec = {
            "image_name": str(relative_image).replace("\\", "/"),
            "humidity": float(row[humidity_col]) if humidity_col and pd.notna(row[humidity_col]) else 0.0,
            "temperature": float(row[temp_col]) if temp_col and pd.notna(row[temp_col]) else 0.0,
            "wind_speed": 0.0,
            "pressure": 0.0,
            "hour": 0.0,
            "latitude": 0.0,
            "longitude": 0.0,
            "pm25": float(row[pm25_col]) if pm25_col and pd.notna(row[pm25_col]) else None,
            "pm10": float(row[pm10_col]) if pm10_col and pd.notna(row[pm10_col]) else None,
            "aqi": float(row[aqi_col]) if aqi_col and pd.notna(row[aqi_col]) else None,
        }
        records.append(rec)

    out_df = pd.DataFrame(records).dropna(subset=["aqi", "pm25", "pm10"], how="all")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Source CSV: {csv_path}")
    print(f"Saved merged AQI metadata: {out_csv} with {len(out_df)} rows")


if __name__ == "__main__":
    main()
