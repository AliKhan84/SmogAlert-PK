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
│   ├── retrain_agent.py      # Auto-retraining orchestrator (evaluates + uploads to R2)
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

## Live Infrastructure (as of June 2026)

| Service | URL / Connection | Status |
|---------|-----------------|--------|
| Railway API | `https://api-production-1df1.up.railway.app` | ✅ Live |
| Vercel Frontend | Deployed on Vercel (auto-deploy from `main`) | ✅ Live |
| Railway PostgreSQL | `postgres.railway.internal:5432` (internal), `acela.proxy.rlwy.net:20010` (public proxy) | ✅ Live |
| Upstash Redis | `useful-flamingo-105053.upstash.io:6379` | ✅ Configured |
| Cloudflare R2 | Bucket `smogalert-pk` | ✅ Configured |
| Supabase | `qykraxdfjrpdrhkoiccd.supabase.co` | ⚠️ Unused (IPv6-only; tables exist but Railway PostgreSQL is primary DB) |

### Railway Project IDs (needed for MCP tools)
- Project: `SmogAlert PK` — `f4887f9f-5f99-4e7e-a5a5-64647b0e7efe`
- API service: `api` — `bbc90dd2-a347-4ba4-8526-c68f42212bd0`
- PostgreSQL service: `Postgres` — `467091ab-20f2-4a46-ad32-29220f2427b1`
- Environment: `production` — `73641369-c9f7-4b36-82f8-161bafd07a86`

### DATABASE_URL (Railway PostgreSQL — primary DB)
```
postgresql+asyncpg://postgres:TZhgSJmaSpleKsCzkpiAWXnDpEmEajWN@postgres.railway.internal:5432/railway
```
For local scripts use the public proxy URL:
```
postgresql+asyncpg://postgres:TZhgSJmaSpleKsCzkpiAWXnDpEmEajWN@acela.proxy.rlwy.net:20010/railway
```

## SaaS Phase Status

| Phase | Description | Status | What's Done | What's Remaining |
|-------|-------------|--------|-------------|-----------------|
| Phase 1 | Repo restructure & monorepo setup | ✅ 100% | Full `apps/`+`packages/`+`agents/` monorepo, CI workflows | — |
| Phase 2 | Live data layer & AI scraping agent | ✅ 90% | `data_scraper.py`, `retrain_agent.py`, `backfill_kaggle_data.py`, ORM models, Celery tasks | Backfill script not yet run; Celery workers not deployed to Railway |
| Phase 3 | Database schema & FastAPI backend | ✅ 90% | All 8 ORM models, all 5 routers (`/auth`,`/users`,`/aqi`,`/alerts`,`/admin`), Redis caching, WebSocket `/aqi/ws`, Railway deployment live | `CLERK_WEBHOOK_SECRET` empty (webhook unverified); `/alerts/test` endpoint stub |
| Phase 4 | User management & alert preferences | 🔄 55% | Clerk middleware, auth webhook (creates user+subscription+prefs), `/users/me` GET+PUT, onboarding page, settings page | WhatsApp opt-in flow, SMS OTP verification, subscription gating middleware |
| Phase 5 | Alert engine (email/WhatsApp/SMS) | 🔄 70% | Full alert engine (`alert_task.py`) with Redis dedup, bilingual messages (EN+UR), `alert_dispatcher.py` with Email/WhatsApp/SMS senders, Resend email HTML template | Twilio credentials empty, Meta WhatsApp token empty, Celery worker not running on Railway |
| Phase 6 | Next.js frontend web app | 🔄 65% | All 7 pages scaffolded (landing, dashboard, alerts, settings, onboarding, admin, sign-in/up), AQI components (Gauge, ForecastChart, PakistanMap, PollutantBar), Clerk auth | Supabase Realtime points at empty Supabase tables (needs fix/removal), PWA manifest, dark mode |
| Phase 7 | ML pipeline integration into API | 🔄 55% | `forecast_service.py`, `model_loader.py`, `/aqi/{city}/forecast` endpoint wired | R2 model upload after retrain, on-demand forecast from local `.pkl` files not tested end-to-end |
| Phase 8 | Deployment (Railway + Vercel + Supabase) | 🔄 70% | Railway API live, Railway PostgreSQL live, Vercel frontend live, Upstash Redis configured, R2 bucket configured, Clerk keys set, Resend key set, WAQI token set | Celery worker + Beat services on Railway, Clerk webhook secret, Twilio/Meta credentials, GitHub Actions CD, custom domain |
| Phase 9 | Mobile app (React Native + Expo) | ⏸️ 0% | — | Future phase after 200+ web users |

## Tech Stack

### Frontend (`apps/web/`)
- **Next.js 14** (App Router) + **Tailwind CSS** + **shadcn/ui**
- **Clerk** for auth (middleware, sign-in/sign-up routes)
- **React Leaflet** for Pakistan AQI map
- **Recharts** for forecast + time-series charts
- **Zustand** for global state, **Supabase Realtime** for live updates (⚠️ currently points at empty Supabase tables — needs data sync or swap to Railway WebSocket)

### Backend (`apps/api/`)
- **FastAPI** with async SQLAlchemy + Alembic migrations
- **Celery** workers: `alert_task`, `retrain_task`, `scrape_task` (defined but not yet deployed as Railway services)
- **Redis** (Upstash) for caching and Celery broker
- Routers: `/auth`, `/users`, `/aqi`, `/alerts`, `/admin`

### Database
- **Primary: Railway PostgreSQL** (`postgres.railway.internal:5432`) — all 8 tables auto-created via SQLAlchemy `create_all` on startup
- **Supabase** (`qykraxdfjrpdrhkoiccd.supabase.co`) — tables exist but unreachable from Railway (IPv6-only free tier). Supabase Realtime subscription in frontend points here (stale).
- Key tables: `users`, `user_locations`, `subscriptions`, `alert_preferences`, `alerts_sent`, `aqi_readings`, `forecasts`, `model_versions`

### Alerts
- **Email**: Resend API (`re_9Z1aasrX_...`) — configured ✅
- **WhatsApp**: Meta Cloud API (WhatsApp Business) — token not yet set ❌
- **SMS**: Twilio — credentials not yet set ❌

### Live Data Sources
- **OpenAQ v3 API** — Lahore, Karachi, Islamabad (free)
- **WAQI API** — all 5 PK cities (token: `b888d3f8...`) ✅
- **OpenMeteo** — fallback for missing cities
- **EPA Pakistan scraper** — Playwright headless (best-effort)

### Deployment
- Frontend: **Vercel** (auto-deploy from `main`)
- Backend API: **Railway** — `api` service
- Background Workers: **Railway** — Celery worker + Beat services (⚠️ not yet deployed)
- Models: **Cloudflare R2** bucket `smogalert-pk`

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
streamlit run dashboard/app.py   # http://localhost:8501
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

### FastAPI backend (local)
```bash
cd apps/api
pip install -r requirements.txt
uvicorn main:app --reload        # http://localhost:8000
```

### Next.js frontend (local)
```bash
cd apps/web
npm install
npm run dev                      # http://localhost:3000
```

### Celery workers (local)
```bash
cd apps/api
celery -A workers.celery_app worker --loglevel=info
celery -A workers.celery_app beat  --loglevel=info
```

### Run backfill (populate Railway DB with Kaggle history)
```bash
# Set DATABASE_URL to Railway public proxy first, then:
cd /home/alikhan/claude-workspace/SmogAlert-PK
DATABASE_URL="postgresql+asyncpg://postgres:TZhgSJmaSpleKsCzkpiAWXnDpEmEajWN@acela.proxy.rlwy.net:20010/railway" \
  python agents/backfill_kaggle_data.py
```

### Run live scraper manually
```bash
python agents/data_scraper.py
```

## Key Design Decisions

- **Monorepo**: `apps/` + `packages/` + `agents/` in one repo — shared CI, easier to wire ML into API.
- **FastAPI not Django**: same Python ecosystem as ML code, async, lighter.
- **Clerk over Supabase Auth**: better DX, pre-built UI, easier phone/WhatsApp collection.
- **Celery over Lambda**: Railway free tier, no cold starts, simpler Beat scheduling.
- **Rule-based source classifier kept**: interpretable, no GPU, good for alert context.
- **Per-city Prophet models**: city-specific seasonality (Lahore smog ≠ Karachi sea breeze).
- **WhatsApp primary channel**: 93% Pakistan penetration; cheaper, richer templates than SMS.
- **Railway PostgreSQL over Supabase**: Supabase free tier is IPv6-only; Railway's native PostgreSQL is on the same private network as the API — zero latency, zero connectivity issues. Supabase tables exist but are not the live database.

## Season Map

```python
Winter: months 11, 12, 1, 2
Spring: months 3, 4, 5
Summer: months 6, 7, 8, 9
Autumn: month 10
```

## Environment Variables

See `.env` at repo root for all credentials. Key variables:

```bash
# Primary DB (Railway PostgreSQL — used in production)
DATABASE_URL=postgresql+asyncpg://postgres:<pass>@postgres.railway.internal:5432/railway

# Supabase (frontend Realtime subscriptions only — NOT the main DB)
SUPABASE_URL=https://qykraxdfjrpdrhkoiccd.supabase.co
SUPABASE_ANON_KEY=...

# Redis (Upstash)
REDIS_URL=rediss://...@useful-flamingo-105053.upstash.io:6379

# Auth (Clerk)
CLERK_SECRET_KEY=sk_test_...
CLERK_WEBHOOK_SECRET=  # ← EMPTY — webhook signature verification disabled

# Email (Resend) — configured ✅
RESEND_API_KEY=re_9Z1aasrX_...

# SMS / WhatsApp — NOT configured ❌
TWILIO_ACCOUNT_SID=
META_WHATSAPP_TOKEN=

# AQI APIs
WAQI_TOKEN=b888d3f8...  # configured ✅
OPENAQ_API_KEY=         # empty — public API works without key

# Model Storage (Cloudflare R2) — configured ✅
CLOUDFLARE_R2_ACCESS_KEY=...
CLOUDFLARE_R2_BUCKET=smogalert-pk
```

## Immediate Next Steps (Priority Order)

1. **Run backfill** — populate `aqi_readings` table with Kaggle history so dashboard shows data
2. **Deploy Celery workers on Railway** — add `worker` and `beat` services so scraper runs hourly
3. **Fix Supabase Realtime** — either sync Railway PostgreSQL → Supabase, or replace frontend Realtime with the WebSocket at `wss://api-production-1df1.up.railway.app/api/aqi/ws`
4. **Set CLERK_WEBHOOK_SECRET** — get from Clerk dashboard → Webhooks, set on Railway API service
5. **Twilio + Meta WhatsApp credentials** — to enable SMS/WhatsApp alert delivery
6. **Test end-to-end forecast** — verify `/api/aqi/{city}/forecast` returns Prophet predictions (needs models uploaded to R2 or loaded from local pkl files)

## Code Style

- Write detailed comments aimed at beginner data-science students
- Use descriptive variable names (`pm25_values`, not `vals`)
- Add docstrings to every function describing parameters and return values
- Print progress messages during long-running operations
