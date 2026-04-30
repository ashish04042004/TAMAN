from __future__ import annotations

import argparse
import io
import os
import zipfile
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a TRAQID subset with aligned metadata.")
    parser.add_argument("--zip-path", default="E:/TRAQID/traqid.zip")
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="data/raw/traqid_subset")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    front_dir = output_root / "front"
    os.makedirs(front_dir, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as zf:
        zip_entries = set(zf.namelist())
        df = pd.read_csv(io.BytesIO(zf.read("TRAQID.csv")))
        df["Image"] = df["Image"].astype(int)
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        selected_rows = []
        for _, row in df.iterrows():
            image_id = int(row["Image"])
            zip_image_path = f"front/{image_id}.png"
            if zip_image_path not in zip_entries:
                continue
            zf.extract(zip_image_path, path=output_root)
            selected_rows.append(row)
            if len(selected_rows) >= args.subset_size:
                break

    if not selected_rows:
        raise RuntimeError("No aligned image-label samples were extracted.")

    subset_df = pd.DataFrame(selected_rows).copy()
    season_map = {"Summer": 0, "Monsoon": 1, "Winter": 2}
    subset_df["image_name"] = subset_df["Image"].apply(lambda x: f"traqid_subset/front/{int(x)}.png")
    subset_df["humidity"] = subset_df["Humidity"]
    subset_df["temperature"] = subset_df["Temperature"]
    subset_df["hour"] = pd.to_datetime(subset_df["created_at"]).dt.hour.astype(float)
    subset_df["is_night"] = (
        subset_df["Day_or_Night"].astype(str).str.strip().str.lower().eq("night").astype(float)
    )
    subset_df["season_code"] = (
        subset_df["Season"].astype(str).str.strip().map(season_map).fillna(0).astype(float)
    )
    subset_df["pm25"] = subset_df["PM2.5"]
    subset_df["pm10"] = subset_df["PM10"]
    subset_df["aqi"] = subset_df["aqi"].astype(float)

    out_cols = [
        "image_name",
        "humidity",
        "temperature",
        "hour",
        "is_night",
        "season_code",
        "pm25",
        "pm10",
        "aqi",
    ]
    out_df = subset_df[out_cols]
    os.makedirs("data/raw", exist_ok=True)
    out_df.to_csv("data/raw/air_quality_metadata.csv", index=False)
    print(f"Extracted samples: {len(out_df)}")
    print("Saved metadata: data/raw/air_quality_metadata.csv")
    print(f"Extracted images root: {output_root}")


if __name__ == "__main__":
    main()
