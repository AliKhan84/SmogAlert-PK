# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmogAlert PK is an air quality monitoring and alert system for Pakistan, built for the **SmogNet Datathon (UET Mardan)**. It implements a 3-stage pipeline — anomaly detection → source classification → alert generation — over a Kaggle dataset covering 5 Pakistani cities and 8 pollutants (Aug 2021 – Dec 2024).

**Full execution plan**: `docs/EXECUTION_PLAN.md` — read this before starting any new phase.

**Phase status**:
- [x] Phase 0 — Dataset acquisition & audit
- [x] Phase 1 — Data layer rebuild (download_data.py + preprocess.py)
- [x] Phase 2 — City + season-aware anomaly detection (model.py)
- [x] Phase 3 — Source classification (source_classifier.py)
- [x] Phase 4 — Enhanced alert generation (alert_system.py)
- [x] Phase 5 — Dashboard upgrade (app.py)
- [x] Phase 6 — End-to-end validation

## Architecture

The project follows a linear pipeline:

```
download_data.py
  → src/preprocess.py
    → src/model.py
      → src/source_classifier.py
        → src/alert_system.py
          → dashboard/app.py
```

1. **Data Acquisition** — `download_data.py`: reads individual city files from `data/Training/` and `data/Testing/`, merges into one combined CSV.
2. **Preprocessing** — `src/preprocess.py`: cleans data, handles 8 pollutants, adds seasonal features, train/test split column.
3. **Anomaly Detection + Forecasting** — `src/model.py`: per-city-season Isolation Forest; Random Forest AQI classifier; Prophet 24-hour PM2.5 forecast.
4. **Source Classification** — `src/source_classifier.py`: rule-based chemical fingerprint classification of anomalies.
5. **Alert Generation** — `src/alert_system.py`: source-driven bilingual (English + Urdu) template alerts.
6. **Dashboard** — `dashboard/app.py`: Streamlit app with map, anomaly timeline, source classification tab, alert table.

## Dataset

**Kaggle**: `hajramohsin/pakistan-air-quality-pollutant-concentrations`

- 5 cities: Islamabad, Karachi, Lahore, Peshawar, Quetta
- 8 pollutants: PM2.5, PM10, NO, NO2, SO2, NH3, CO, O3
- Training: Aug 2021 – Jun 2024 (`data/Training/`) — 103,794 rows after cleaning
- Testing: Jul – Dec 2024 (`data/Testing/`) — 23,757 rows after cleaning
- Split boundary: `2024-07-01`

**File formats** (important — affects date parsing):
- `data/Training/*.xlsx` (Islamabad, Karachi, Lahore): dates as `YYYY-MM-DD`
- `data/Training/*.csv` (Peshawar, Quetta): dates as `DD/MM/YYYY`
- `data/Testing/*.csv` (all cities): dates as `D/M/YYYY`
- All parsed with `dayfirst=True` in `download_data.py` and normalised to ISO format before saving.

## Data Flow

| Path | Description |
|------|-------------|
| `data/Training/` | 5 city files (3 xlsx + 2 csv), Aug 2021 – Jun 2024 |
| `data/Testing/` | 5 city CSV files, Jul – Dec 2024 |
| `data/raw_aqi_data.csv` | Merged raw data, 162,993 rows, 5 cities |
| `data/pakistan_aq_raw.csv` | Canonical copy of merged raw data |
| `data/cleaned_data.csv` | Preprocessed data, 127,551 rows, 20 columns incl. season, city_season, split |
| `models/random_forest_model.pkl` | Trained Random Forest AQI classifier |
| `models/isolation_forest_{city_season}.pkl` | One Isolation Forest model per city-season group (20 files) |
| `models/prophet_model.pkl` | Trained Prophet model for 24-hour PM2.5 forecasting |
| `outputs/anomalies.csv` | Detected anomalous readings, 5,765 rows, all 8 pollutants |
| `outputs/anomalies_classified.csv` | Anomalies + source_label, source_confidence, pollutant_signature |
| `outputs/alerts_log.csv` | Structured bilingual alerts, test-window Unhealthy/Hazardous anomalies |
| `outputs/forecast_24h.csv` | 24-hour PM2.5 forecast values with confidence intervals |
| `outputs/anomaly_plot.png` | Anomaly timeline visualisation |
| `outputs/confusion_matrix.png` | Random Forest classifier confusion matrix |
| `outputs/feature_importance.png` | Random Forest feature importance chart |
| `outputs/forecast_plot.png` | Prophet 24-hour PM2.5 forecast plot |

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Full Pipeline (run in order)
```bash
python download_data.py           # Merge all city files → data/raw_aqi_data.csv
python src/preprocess.py          # Clean + engineer features → data/cleaned_data.csv
python src/model.py               # Anomaly detection + RF classifier → outputs/anomalies.csv
python src/source_classifier.py   # Source classification → outputs/anomalies_classified.csv
python src/alert_system.py        # Alert generation → outputs/alerts_log.csv
```

### Dashboard
```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

## Key Design Decisions

- **Source classifier is rule-based** (not ML): chemical fingerprint thresholds — fast, interpretable, datathon-friendly.
- **Anomaly detection is per-city-season**: one Isolation Forest per `city_season` group (e.g., `Lahore_Winter`), trained on training split only. Contamination = 5%.
- **Anomaly features**: `pm25`, `pm10`, `pm25_24h_avg`, `hour`.
- **Alert criteria**: test split only, AQI level Unhealthy or Hazardous.
- **Both English and Urdu** alert text are kept for all alert records.
- **Random Forest AQI classifier** is retained alongside the anomaly pipeline (doesn't conflict; adds value for dashboard).
- **Prophet forecasting** runs as Part 3 of `model.py`: trained on all historical PM2.5 data, forecasts next 24 hours with confidence intervals. Model saved to `models/prophet_model.pkl`.

## Season Map

```python
Winter: months 11, 12, 1, 2
Spring: months 3, 4, 5
Summer: months 6, 7, 8, 9
Autumn: month 10
```

## Key Dependencies

| Category | Libraries |
|----------|-----------|
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (RandomForest, IsolationForest), prophet |
| Model persistence | joblib |
| Visualisation | matplotlib, seaborn, plotly |
| Geospatial | folium, streamlit-folium |
| Dashboard | streamlit |

## Code Style

- Write detailed comments aimed at beginner data-science students
- Use descriptive variable names (`pm25_values`, not `vals`)
- Add docstrings to every function describing parameters and return values
- Print progress messages during long-running operations
