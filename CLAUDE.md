# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmogAlert PK is an air quality monitoring and alert system for Pakistan. It started as a Streamlit ML demo for the **SmogNet Datathon (UET Mardan)** and is now being upgraded into a **production SaaS** with real-time AQI monitoring, predictive alerts, SMS/WhatsApp delivery, and an AI-driven live data pipeline.

**Full SaaS upgrade plan**: `plan.md` — read this before starting any new phase.

## Current Architecture (Monorepo)

```
SmogAlert-PK/
├── apps/
│   ├── web/          # Next.js 14 frontend (Tailwind + shadcn/ui + Clerk auth)
│   └── api/          # FastAPI backend (SQLAlchemy + Alembic + Celery + Redis)
├── packages/
│   └── ml/           # Existing ML pipeline (moved from src/ + dashboard/)
├── agents/
│   ├── data_scraper.py       # Live AQI scraper (OpenAQ + WAQI + OpenMeteo + EPA)
│   └── backfill_kaggle_data.py  # One-time backfill of Kaggle data into DB
├── data/             # Raw + cleaned CSV data (static Kaggle dataset)
├── models/           # Trained .pkl files (Isolation Forest, RandomForest, Prophet)
├── outputs/          # Pipeline output CSVs and plots
├── docs/
├── .github/
│   └── workflows/    # ci.yml (lint + test) + cron.yml (daily retrain)
├── plan.md           # Full SaaS phase plan
└── CLAUDE.md
```

## SaaS Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Repo restructure & monorepo setup | ~90% done |
| Phase 2 | Live data layer & AI scraping agent | ~60% done |
| Phase 3 | Database schema & FastAPI backend | ~70% done |
| Phase 4 | User management & alert preferences | ~40% done |
| Phase 5 | Alert engine (email/WhatsApp/SMS) | ~40% done |
| Phase 6 | Next.js frontend web app | ~60% done |
| Phase 7 | ML pipeline integration into API | ~40% done |
| Phase 8 | Deployment (Railway + Vercel + Supabase) | ~10% done |
| Phase 9 | Mobile app (React Native + Expo) | 0% — future |

## Tech Stack

### Frontend (`apps/web/`)
- **Next.js 14** (App Router) + **Tailwind CSS** + **shadcn/ui**
- **Clerk** for auth (middleware, sign-in/sign-up routes)
- **React Leaflet** for Pakistan AQI map
- **Recharts** for forecast + time-series charts
- **Zustand** for global state, **Supabase Realtime** for live updates

### Backend (`apps/api/`)
- **FastAPI** with async SQLAlchemy + Alembic migrations
- **Celery** workers: `alert_task`, `retrain_task`, `scrape_task`
- **Redis** (Upstash) for caching and Celery broker
- Routers: `/auth`, `/users`, `/aqi`, `/alerts`, `/admin`

### Database
- **PostgreSQL** via **Supabase** (free tier)
- Key tables: `users`, `user_locations`, `subscriptions`, `alert_preferences`, `alerts_sent`, `aqi_readings`, `forecasts`, `model_versions`

### Alerts
- **Email**: Resend API
- **WhatsApp**: Meta Cloud API (WhatsApp Business)
- **SMS**: Twilio

### Live Data Sources
- **OpenAQ v3 API** — Lahore, Karachi, Islamabad (free)
- **WAQI API** — all 5 PK cities (free key)
- **OpenMeteo** — fallback for missing cities
- **EPA Pakistan scraper** — Playwright headless (best-effort)

### Deployment
- Frontend: **Vercel**
- Backend + Celery workers: **Railway** ($5/month free credit)
- Models: **Cloudflare R2** (10 GB free)

## ML Pipeline (Legacy — now in `packages/ml/`)

The original datathon pipeline is preserved and powers the API's forecast endpoint.

```
packages/ml/download_data.py
  → packages/ml/src/preprocess.py
    → packages/ml/src/model.py          # Isolation Forest + RF + Prophet
      → packages/ml/src/source_classifier.py
        → packages/ml/src/alert_system.py
          → packages/ml/dashboard/app.py  # Legacy Streamlit dashboard
```

**Run legacy pipeline** (still works from repo root):
```bash
python download_data.py
python src/preprocess.py
python src/model.py
python src/source_classifier.py
python src/alert_system.py
streamlit run dashboard/app.py
```

## Dataset

**Kaggle**: `hajramohsin/pakistan-air-quality-pollutant-concentrations`

- 5 cities: Islamabad, Karachi, Lahore, Peshawar, Quetta
- 8 pollutants: PM2.5, PM10, NO, NO2, SO2, NH3, CO, O3
- Training: Aug 2021 – Jun 2024 (`data/Training/`) — 103,794 rows
- Testing: Jul – Dec 2024 (`data/Testing/`) — 23,757 rows
- Split boundary: `2024-07-01`

**File formats** (affects date parsing):
- `data/Training/*.xlsx` (Islamabad, Karachi, Lahore): `YYYY-MM-DD`
- `data/Training/*.csv` (Peshawar, Quetta): `DD/MM/YYYY`
- `data/Testing/*.csv` (all cities): `D/M/YYYY`
- All parsed with `dayfirst=True`, normalised to ISO before saving.

## Data Flow

| Path | Description |
|------|-------------|
| `data/Training/` | 5 city files (3 xlsx + 2 csv) |
| `data/Testing/` | 5 city CSV files |
| `data/raw_aqi_data.csv` | Merged raw data, 162,993 rows |
| `data/cleaned_data.csv` | Preprocessed, 127,551 rows, 20 columns |
| `models/random_forest_model.pkl` | Trained RF AQI classifier |
| `models/isolation_forest_{city_season}.pkl` | 20 Isolation Forest models |
| `models/prophet_{city}.pkl` | Per-city Prophet PM2.5 forecast models |
| `outputs/anomalies.csv` | Detected anomalies, 5,765 rows |
| `outputs/anomalies_classified.csv` | Anomalies + source labels |
| `outputs/alerts_log.csv` | Bilingual alerts (Unhealthy/Hazardous only) |
| `outputs/forecast_24h.csv` | 24-hour PM2.5 forecast with CI bands |

## Development Commands

### Legacy ML pipeline
```bash
pip install -r requirements.txt
python download_data.py
python src/preprocess.py
python src/model.py
python src/source_classifier.py
python src/alert_system.py
streamlit run dashboard/app.py   # http://localhost:8501
```

### FastAPI backend
```bash
cd apps/api
pip install -r requirements.txt
uvicorn main:app --reload        # http://localhost:8000
```

### Next.js frontend
```bash
cd apps/web
npm install
npm run dev                      # http://localhost:3000
```

### Celery workers
```bash
cd apps/api
celery -A workers.celery_app worker --loglevel=info
celery -A workers.celery_app beat  --loglevel=info
```

## Key Design Decisions

- **Monorepo**: `apps/` + `packages/` + `agents/` in one repo — shared CI, easier to wire ML into API.
- **FastAPI not Django**: same Python ecosystem as ML code, async, lighter.
- **Clerk over Supabase Auth**: better DX, pre-built UI, easier phone/WhatsApp collection.
- **Celery over Lambda**: Railway free tier, no cold starts, simpler Beat scheduling.
- **Rule-based source classifier kept**: interpretable, no GPU, good for alert context.
- **Per-city Prophet models**: city-specific seasonality (Lahore smog ≠ Karachi sea breeze).
- **WhatsApp primary channel**: 93% Pakistan penetration; cheaper, richer templates than SMS.

## Season Map

```python
Winter: months 11, 12, 1, 2
Spring: months 3, 4, 5
Summer: months 6, 7, 8, 9
Autumn: month 10
```

## Environment Variables

See `.env.example` at repo root for all required variables (Supabase, Redis, Clerk, Resend, Twilio, Meta WhatsApp, WAQI, OpenAQ, Cloudflare R2).

## Code Style

- Write detailed comments aimed at beginner data-science students
- Use descriptive variable names (`pm25_values`, not `vals`)
- Add docstrings to every function describing parameters and return values
- Print progress messages during long-running operations
