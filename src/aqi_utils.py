from __future__ import annotations

import math


def _calc_sub_index(
    concentration: float,
    bp_lo: float,
    bp_hi: float,
    i_lo: int,
    i_hi: int,
) -> float:
    return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo


def _pm25_sub_index(pm25: float) -> float:
    # US EPA AQI breakpoints for PM2.5 (24-hour average)
    if pm25 <= 12.0:
        return _calc_sub_index(pm25, 0.0, 12.0, 0, 50)
    if pm25 <= 35.4:
        return _calc_sub_index(pm25, 12.1, 35.4, 51, 100)
    if pm25 <= 55.4:
        return _calc_sub_index(pm25, 35.5, 55.4, 101, 150)
    if pm25 <= 150.4:
        return _calc_sub_index(pm25, 55.5, 150.4, 151, 200)
    if pm25 <= 250.4:
        return _calc_sub_index(pm25, 150.5, 250.4, 201, 300)
    if pm25 <= 350.4:
        return _calc_sub_index(pm25, 250.5, 350.4, 301, 400)
    if pm25 <= 500.4:
        return _calc_sub_index(pm25, 350.5, 500.4, 401, 500)
    return 500.0


def _pm10_sub_index(pm10: float) -> float:
    # US EPA AQI breakpoints for PM10 (24-hour average)
    if pm10 <= 54:
        return _calc_sub_index(pm10, 0, 54, 0, 50)
    if pm10 <= 154:
        return _calc_sub_index(pm10, 55, 154, 51, 100)
    if pm10 <= 254:
        return _calc_sub_index(pm10, 155, 254, 101, 150)
    if pm10 <= 354:
        return _calc_sub_index(pm10, 255, 354, 151, 200)
    if pm10 <= 424:
        return _calc_sub_index(pm10, 355, 424, 201, 300)
    if pm10 <= 504:
        return _calc_sub_index(pm10, 425, 504, 301, 400)
    if pm10 <= 604:
        return _calc_sub_index(pm10, 505, 604, 401, 500)
    return 500.0


def compute_aqi_from_pollutants(pm25: float | None = None, pm10: float | None = None) -> int:
    sub_indices = []
    if pm25 is not None and not math.isnan(pm25):
        sub_indices.append(_pm25_sub_index(float(pm25)))
    if pm10 is not None and not math.isnan(pm10):
        sub_indices.append(_pm10_sub_index(float(pm10)))
    if not sub_indices:
        raise ValueError("At least one pollutant value (pm25 or pm10) is required to compute AQI.")
    return int(round(max(sub_indices)))
