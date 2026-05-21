# SmogAlert PK — Air Quality Monitoring & Alert System

An end-to-end air quality monitoring pipeline for Pakistan, built for the **SmogNet Datathon (UET Mardan)**. Detects anomalous pollution events, classifies their pollution sources, generates bilingual alerts, forecasts 24-hour PM2.5 levels, and presents everything through an interactive Streamlit dashboard.

---

## Pipeline Overview

```
download_data.py          →  Merge 5-city Kaggle files into one dataset
src/preprocess.py         →  Clean, engineer features, seasonal labelling
src/model.py              →  Anomaly detection + RF classifier + Prophet forecast
src/source_classifier.py  →  Rule-based pollution source classification
src/alert_system.py       →  Bilingual (English + Urdu) alert generation
dashboard/app.py          →  Streamlit dashboard
```

**3-stage core pipeline:**
1. **Anomaly Detection** — per-city-season Isolation Forest flags unusual pollution spikes
2. **Source Classification** — chemical fingerprint rules identify the likely pollution source (vehicle emissions, industrial, dust storms, crop burning, etc.)
3. **Alert Generation** — source-driven bilingual templates produce structured alerts for Unhealthy/Hazardous readings

Plus a **Prophet 24-hour PM2.5 forecast** with confidence intervals.

---

## Dataset

**Kaggle**: [`hajramohsin/pakistan-air-quality-pollutant-concentrations`](https://www.kaggle.com/datasets/hajramohsin/pakistan-air-quality-pollutant-concentrations)

| | Details |
|---|---|
| Cities | Islamabad, Karachi, Lahore, Peshawar, Quetta |
| Pollutants | PM2.5, PM10, NO, NO2, SO2, NH3, CO, O3 |
| Training period | Aug 2021 – Jun 2024 (103,794 rows) |
| Testing period | Jul – Dec 2024 (23,757 rows) |

---

## Project Structure

```
SmogAlert-PK/
├── data/
│   ├── Training/          # 5 city source files (3 .xlsx + 2 .csv), Aug 2021–Jun 2024
│   └── Testing/           # 5 city CSV files, Jul–Dec 2024
├── src/
│   ├── preprocess.py      # Data cleaning, feature engineering, season labelling
│   ├── model.py           # Isolation Forest + Random Forest + Prophet forecasting
│   ├── source_classifier.py  # Rule-based pollution source classification
│   └── alert_system.py    # Bilingual alert generation
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── models/
│   ├── isolation_forest_{city_season}.pkl  # 20 per-city-season models
│   ├── random_forest_model.pkl             # AQI level classifier
│   └── prophet_model.pkl                   # 24-hour PM2.5 forecast model
├── outputs/
│   ├── anomalies.csv              # 5,765 detected anomalous readings
│   ├── anomalies_classified.csv   # Anomalies + source labels
│   ├── alerts_log.csv             # Structured bilingual alerts
│   ├── forecast_24h.csv           # Prophet 24-hour forecast values
│   ├── anomaly_plot.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── forecast_plot.png
├── docs/
│   └── EXECUTION_PLAN.md
├── download_data.py
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run scripts in order — each step produces the inputs for the next:

```bash
python download_data.py           # → data/raw_aqi_data.csv
python src/preprocess.py          # → data/cleaned_data.csv
python src/model.py               # → outputs/anomalies.csv + models/
python src/source_classifier.py   # → outputs/anomalies_classified.csv
python src/alert_system.py        # → outputs/alerts_log.csv
```

The `outputs/` and `models/` files are already included in this repo — you can skip straight to the dashboard without re-running the pipeline.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Features include:

- Interactive map of anomaly locations by city
- Anomaly timeline by pollutant and season
- Source classification breakdown
- Bilingual alert table (English + Urdu)
- 24-hour PM2.5 forecast with confidence band

---

## Key Design Decisions

- **Anomaly detection is per-city-season**: one Isolation Forest per `{city}_{season}` group (20 models total), trained only on the training split. Contamination = 5%.
- **Source classifier is rule-based**: chemical fingerprint thresholds rather than ML — fast, interpretable, and datathon-friendly.
- **Alert criteria**: test split only, AQI level Unhealthy or Hazardous.
- **Bilingual alerts**: both English and Urdu text are kept for every alert record.
- **Prophet forecasting**: trained on all historical PM2.5 data; outputs next 24 hours with upper/lower confidence intervals.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (IsolationForest, RandomForest), prophet |
| Model persistence | joblib |
| Visualisation | matplotlib, seaborn, plotly |
| Geospatial | folium, streamlit-folium |
| Dashboard | streamlit |
