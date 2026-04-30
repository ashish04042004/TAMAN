# Report Citation Notes (Ground Truth + Data Sources)

Use this section in your final report to justify data credibility.

## Suggested Citable Sources

1. **OpenAQ (air-quality pollutant observations)**
   - Website: https://openaq.org/
   - API docs: https://docs.openaq.org/
   - Why cite: Pollutant values (e.g., PM2.5, PM10) are measured by real monitoring stations and published openly.

2. **Outdoor Webcam Image Source (choose one open source)**
   - Example family: AMOS/public outdoor camera datasets from research institutions.
   - Why cite: Timestamped location-aware images can be aligned with sensor observations.

3. **Meteorological Variables**
   - If fetched from OpenAQ linked data or public weather APIs, cite that provider docs.
   - Why cite: Humidity/temperature/wind/pressure are standard measured variables.

2. **AQI Standard Formula**
   - US EPA AQI basics: https://www.airnow.gov/aqi/aqi-basics/
   - Why cite: AQI target is deterministically computed from pollutant concentrations via published breakpoints.

## Ground Truth Statement Template

"Ground truth AQI labels are computed from public monitoring-station pollutant measurements (PM2.5/PM10) available via OpenAQ API, using published AQI breakpoint equations.  
Each label corresponds to the nearest station-time observation aligned with image capture timestamp and location."

## Verifiability Checklist

- Include dataset version/date of download.
- Include API endpoint and access date.
- Keep raw source dump snapshots or exported CSVs.
- Include matching logic (time window and distance threshold) in methodology.
