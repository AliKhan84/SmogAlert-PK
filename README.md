# SmogAlert PK

**Real-time air quality alerts for Pakistan — know before you breathe.**

SmogAlert PK monitors air quality across 5 Pakistani cities in real time, detects pollution anomalies, forecasts the next 24 hours of PM2.5 levels, and delivers bilingual (English + Urdu) alerts via WhatsApp, SMS, and email — before exposure happens.

**Live:** [smogalert-pk.vercel.app](https://smogalert-pk.vercel.app) · **API:** [api-production-1df1.up.railway.app](https://api-production-1df1.up.railway.app/docs) · **Repo:** [github.com/AliKhan84/SmogAlert-PK](https://github.com/AliKhan84/SmogAlert-PK)

---

## Inspiration

Pakistan is home to some of the most polluted cities on Earth. Lahore regularly tops global AQI rankings, and during winter smog season PM2.5 concentrations routinely exceed **500 µg/m³** — over 20× the WHO safe limit of 15 µg/m³. Existing solutions are either paywalled dashboards built for researchers or government portals that are rarely updated.

We wanted proactive alerts pushed directly to WhatsApp — the one app with 93% penetration in Pakistan — in both English and Urdu, before exposure happens.

---

## What it does

- **Live AQI dashboard** — real-time pollutant readings on an interactive Pakistan map
- **24-hour PM2.5 forecasts** — per-city Prophet models with 95% confidence bands
- **Anomaly detection** — Isolation Forest flags unusual pollution spikes as they happen
- **Bilingual alerts** — English + Urdu notifications via WhatsApp, SMS, and email
- **User preferences** — set your city, alert threshold, and preferred delivery channel

---

## Architecture

```
SmogAlert-PK/
├── apps/
│   ├── web/          # Next.js 14 frontend (Tailwind + shadcn/ui + Clerk auth)
│   └── api/          # FastAPI backend (SQLAlchemy + Alembic + Celery + Redis)
├── packages/
│   └── ml/           # ML pipeline (Isolation Forest + Random Forest + Prophet)
├── agents/
│   ├── data_scraper.py         # Live AQI scraper (OpenAQ + WAQI + OpenMeteo)
│   ├── retrain_agent.py        # Auto-retraining orchestrator
│   └── backfill_kaggle_data.py # One-time historical backfill
├── data/             # Raw + cleaned CSV/XLSX (Kaggle dataset)
├── models/           # Trained .pkl files
└── outputs/          # Pipeline output CSVs and plots
```

---

## ML Pipeline

**4-stage pipeline:**

1. **Anomaly Detection** — 20 per-city-season Isolation Forest models (contamination = 5%) flag unusual pollution spikes
2. **Source Classification** — chemical fingerprint rules identify the likely pollution source (vehicular, industrial, dust storms, crop burning)
3. **Alert Generation** — source-driven bilingual templates produce structured alerts for Unhealthy/Hazardous readings
4. **24h PM2.5 Forecasting** — one Prophet model per city with 95% confidence intervals; forward-looking alerts fire on predicted threshold breaches

AQI is computed from raw PM2.5 using the US EPA piecewise linear formula:

```
I = ((I_hi - I_lo) / (C_hi - C_lo)) * (C - C_lo) + I_lo
```

---

## Dataset

**Kaggle:** [`hajramohsin/pakistan-air-quality-pollutant-concentrations`](https://www.kaggle.com/datasets/hajramohsin/pakistan-air-quality-pollutant-concentrations)

| | Details |
|---|---|
| Cities | Islamabad, Karachi, Lahore, Peshawar, Quetta |
| Pollutants | PM2.5, PM10, NO, NO₂, SO₂, NH₃, CO, O₃ |
| Training | Aug 2021 – Jun 2024 — 103,794 rows |
| Testing | Jul – Dec 2024 — 23,757 rows |

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python, TypeScript, SQL |
| **Frontend** | Next.js 14, Tailwind CSS, shadcn/ui, React Leaflet, Recharts, Zustand |
| **Backend** | FastAPI, Celery, SQLAlchemy, Alembic |
| **ML** | scikit-learn (IsolationForest, RandomForest), Prophet, pandas, numpy |
| **Database** | PostgreSQL (Railway), Redis (Upstash) |
| **Auth** | Clerk |
| **Alerts** | Resend (email), Twilio (SMS), Meta WhatsApp Business API |
| **Data APIs** | OpenAQ v3, WAQI, OpenMeteo |
| **Cloud** | Railway (API + workers + DB), Vercel (frontend), Cloudflare R2 (models) |
| **DevOps** | GitHub Actions, Nixpacks |

---

## Live Infrastructure

| Service | URL |
|---|---|
| Frontend | [smogalert-pk.vercel.app](https://smogalert-pk.vercel.app) |
| API | [api-production-1df1.up.railway.app](https://api-production-1df1.up.railway.app) |
| API Docs | [api-production-1df1.up.railway.app/docs](https://api-production-1df1.up.railway.app/docs) |

---

## Running Locally

**FastAPI backend**
```bash
cd apps/api
pip install -r requirements.txt
uvicorn main:app --reload
```

**Next.js frontend**
```bash
cd apps/web
npm install
npm run dev
```

**Legacy ML pipeline**
```bash
pip install -r requirements.txt
python download_data.py
python src/preprocess.py
python src/model.py
python src/source_classifier.py
python src/alert_system.py
streamlit run dashboard/app.py
```

---

## Team

- **Ali Khan**
- **M Sudais**
- **Younas Khan**
