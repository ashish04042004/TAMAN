"""
Build a metadata file by pairing local images with OpenAQ measurements.

Expected input CSV: data/raw/image_index.csv
Columns:
    image_name,timestamp_utc,latitude,longitude

Output CSV: data/raw/air_quality_metadata.csv
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from tqdm import tqdm

from aqi_utils import compute_aqi_from_pollutants

OPENAQ_LATEST_URL = "https://api.openaq.org/v3/sensors/{sensor_id}/measurements"


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_measurements(sensor_id: int, start: datetime, end: datetime, api_key: str | None = None):
    headers = {"X-API-Key": api_key} if api_key else {}
    params = {
        "date_from": to_iso(start),
        "date_to": to_iso(end),
        "limit": 1000,
    }
    url = OPENAQ_LATEST_URL.format(sensor_id=sensor_id)
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def choose_nearest(record_time: datetime, measurements: list[dict]):
    if not measurements:
        return None
    best = None
    best_delta = None
    for m in measurements:
        observed = datetime.fromisoformat(m["period"]["datetimeFrom"]["utc"].replace("Z", "+00:00"))
        delta = abs((observed - record_time).total_seconds())
        if best is None or delta < best_delta:
            best = m
            best_delta = delta
    return best


def main():
    api_key = os.getenv("OPENAQ_API_KEY")
    sensor_id = int(os.getenv("OPENAQ_SENSOR_ID", "0"))
    if sensor_id == 0:
        raise ValueError("Set OPENAQ_SENSOR_ID environment variable to a valid sensor id.")

    index_csv = "data/raw/image_index.csv"
    out_csv = "data/raw/air_quality_metadata.csv"
    df = pd.read_csv(index_csv)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pairing records"):
        ts = datetime.fromisoformat(str(row["timestamp_utc"]).replace("Z", "+00:00"))
        start = ts - timedelta(hours=1)
        end = ts + timedelta(hours=1)
        measurements = fetch_measurements(sensor_id=sensor_id, start=start, end=end, api_key=api_key)
        nearest = choose_nearest(ts, measurements)
        if nearest is None:
            continue

        pm25 = nearest.get("value", None)
        pm10 = nearest.get("pm10", None)
        if pm25 is None and pm10 is None:
            continue

        try:
            aqi = compute_aqi_from_pollutants(pm25=pm25, pm10=pm10)
        except ValueError:
            continue

        rows.append(
            {
                "image_name": row["image_name"],
                "humidity": nearest.get("humidity", None),
                "temperature": nearest.get("temperature", None),
                "wind_speed": nearest.get("wind", None),
                "pressure": nearest.get("pressure", None),
                "hour": ts.hour,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "pm25": pm25,
                "pm10": pm10,
                "aqi": aqi,
            }
        )

    merged = pd.DataFrame(rows).dropna()
    merged.to_csv(out_csv, index=False)
    print(f"Saved {len(merged)} rows to {out_csv}")


if __name__ == "__main__":
    main()
