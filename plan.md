# SmogAlert PK — SaaS Upgrade Plan

**Objective**: Transform SmogAlert PK from a Streamlit ML demo into a production-grade SaaS application with user accounts, real-time AQI monitoring, 24-hour predictive alerts, SMS/WhatsApp delivery, and an AI-driven self-updating data pipeline.

**Current state**: Streamlit dashboard + offline ML pipeline over static Kaggle dataset (Aug 2021 – Dec 2024).  
**Target state**: Scalable web app serving thousands of users with live data, predictive alerts, and automated model retraining.

---

## Recommended Tech Stack

### Frontend — Web
| Layer | Choice | Why |
|-------|--------|-----|
| Framework | **Next.js 14** (App Router) | SSR/SSG, SEO-friendly, scales to millions, pairs naturally with Vercel free tier |
| Styling | **Tailwind CSS** | Utility-first, fast iteration, mobile-first |
| Components | **shadcn/ui** | Accessible, unstyled Radix primitives styled with Tailwind |
| Maps | **React Leaflet** | Interactive Pakistan AQI overlay, free, open source |
| Charts | **Recharts** | Composable React chart library, AQI time-series and forecasts |
| Auth client | **Clerk** | Drop-in auth UI, generous free tier (10K MAU), email + OAuth |
| State | **Zustand** | Lightweight global state for AQI feeds and user prefs |
| Real-time | **Supabase Realtime** | WebSocket subscriptions for live AQI updates |

### Backend API
| Layer | Choice | Why |
|-------|--------|-----|
| Framework | **FastAPI** (Python) | Same ecosystem as existing ML code, async, auto OpenAPI docs |
| Task queue | **Celery + Redis** | Background alert processing, scheduled scraping, model retraining |
| Scheduler | **Celery Beat** | Cron-like task scheduling (hourly scrape, daily retrain) |
| Cache | **Redis** (Upstash) | AQI data cache, rate limiting, pub/sub for real-time push |
| ORM | **SQLAlchemy + Alembic** | DB migrations, async queries |

### Database
| Layer | Choice | Why |
|-------|--------|-----|
| Primary DB | **PostgreSQL** via **Supabase** | 500 MB free tier, built-in Auth, Row Level Security, Realtime |
| Object storage | **Cloudflare R2** | 10 GB free — stores trained model `.pkl` files between retraining runs |

### Alerts & Messaging
| Channel | Service | Notes |
|---------|---------|-------|
| Email | **Resend** | 3,000 emails/month free, excellent deliverability, React Email templates |
| WhatsApp | **Meta Cloud API (WhatsApp Business)** | 1,000 free user-initiated conversations/month; requires Meta Business verification |
| SMS | **Twilio** | $15 free trial credit; pay-as-you-go after. For Pakistan: ~$0.006/SMS |
| Fallback SMS | **Vonage (Nexmo)** | Alternative if Twilio coverage gaps in PK |

### Live AQI Data Sources (replacing static Kaggle dataset)
| Source | Coverage | Cost |
|--------|----------|------|
| **OpenAQ API v3** | Lahore, Karachi, Islamabad — hourly PM2.5/PM10/O3 | Free, no key needed |
| **WAQI (aqicn.org) API** | All 5 PK cities — AQI, PM2.5, NO2, SO2, CO | Free API key, 1,000 req/day |
| **EPA Pakistan web scraper** | Official PK government data | Scraped via Playwright (headless) |
| **Fallback** | OpenMeteo Air Quality API | Free, global, includes Pakistan |

### Deployment (All Free Tiers)
| Service | Platform | Free Limit |
|---------|----------|------------|
| Frontend | **Vercel** | Unlimited deployments, 100 GB bandwidth/month |
| Backend API | **Railway** | $5/month free credit (~500 hrs compute) |
| Background Workers (Celery) | **Railway** | Same $5 credit, runs as separate service |
| Database | **Supabase** | 500 MB PostgreSQL, 2 GB file storage |
| Redis | **Upstash** | 10,000 commands/day free |
| Model Storage | **Cloudflare R2** | 10 GB free |
| CI/CD + Cron Jobs | **GitHub Actions** | 2,000 minutes/month free |
| Domain | **Freenom** (.tk/.ml) | Free subdomain OR `.pk` TLD ~$10/yr |

### Future Mobile App
- **React Native + Expo** — code-share business logic with Next.js
- **Expo Push Notifications** — free for unlimited devices
- **Firebase FCM** — fallback push delivery

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│  OpenAQ API → WAQI API → EPA Scraper → OpenMeteo API       │
│              ↓  (Celery Beat — hourly)                      │
│         ai_data_agent.py (normalise + store)                │
│              ↓                                              │
│         PostgreSQL: aqi_readings table                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                     ML PIPELINE                             │
│   preprocess.py → model.py (Prophet + IF + RF)             │
│   Auto-retrain: daily via Celery Beat                       │
│   Models saved to Cloudflare R2                             │
│   Forecast served via /api/forecast/{city}                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  BACKEND API (FastAPI)                      │
│  /auth  /users  /aqi  /forecast  /alerts  /subscriptions   │
│  Celery workers: alert_engine, retrain_agent, scrape_agent  │
└──────────┬──────────────────────┬───────────────────────────┘
           │                      │
┌──────────▼──────┐    ┌──────────▼──────────────────────────┐
│  FRONTEND       │    │        ALERT DELIVERY               │
│  Next.js 14     │    │  Email → Resend                     │
│  Vercel         │    │  WhatsApp → Meta Cloud API          │
│  Auth: Clerk    │    │  SMS → Twilio                       │
└─────────────────┘    └─────────────────────────────────────┘
```

---

## Phases

---

### Phase 1 — Repository Restructure & Monorepo Setup
**Goal**: Reorganise the codebase into a production monorepo without breaking the existing ML pipeline.

**New folder structure**:
```
SmogAlert-PK/
├── apps/
│   ├── web/                  # Next.js frontend
│   └── api/                  # FastAPI backend
├── packages/
│   └── ml/                   # Existing ML code (preprocess, model, alert_system etc.)
├── agents/
│   ├── data_scraper.py       # AI data agent (OpenAQ + WAQI + EPA)
│   └── retrain_agent.py      # Auto-retraining orchestrator
├── data/                     # Kept as-is
├── models/                   # Kept as-is
├── outputs/                  # Kept as-is
├── docs/
├── plan.md
└── CLAUDE.md
```

**Tasks**:
- [ ] Move existing `src/`, `dashboard/` into `packages/ml/`
- [ ] Create `apps/web/` scaffold with Next.js 14
- [ ] Create `apps/api/` scaffold with FastAPI
- [ ] Set up shared `.env.example` with all required env vars
- [ ] Configure GitHub Actions CI (lint + test on PR)

---

### Phase 2 — Live Data Layer & AI Scraping Agent
**Goal**: Replace static Kaggle CSV with a continuously updated live feed from multiple APIs.

**`agents/data_scraper.py`** — autonomous data agent:
1. **OpenAQ adapter**: fetch last 24h readings for Lahore, Karachi, Islamabad via OpenAQ v3 API
2. **WAQI adapter**: fetch current AQI + pollutants for all 5 PK cities (free API key)
3. **OpenMeteo adapter**: fallback hourly PM2.5/PM10/NO2 for any missing city
4. **EPA Pakistan scraper**: Playwright headless scrape of official EPA dashboard (best-effort)
5. **Normaliser**: maps all sources to unified schema `{city, timestamp, pm25, pm10, no2, so2, co, o3, source}`
6. **Deduplication**: upsert by `(city, timestamp)` to avoid duplicate rows
7. **Trigger**: runs every hour via Celery Beat schedule

**`agents/retrain_agent.py`** — model retraining orchestrator:
1. Check if ≥500 new rows accumulated since last retrain
2. Pull latest data from PostgreSQL
3. Run `preprocess.py` → `model.py` (Prophet + IsolationForest + RandomForest)
4. Evaluate: if new Prophet MAE < current MAE, promote model; else keep old
5. Upload new `.pkl` files to Cloudflare R2
6. Log retrain event in `model_versions` DB table
7. **Trigger**: runs daily at 2 AM PKT via Celery Beat

**Database tables added**:
- `aqi_readings(id, city, timestamp, pm25, pm10, no2, so2, co, o3, aqi_calculated, source, created_at)`
- `model_versions(id, model_type, city, mae, trained_at, promoted, r2_key)`

**Tasks**:
- [ ] Register free WAQI API key at aqicn.org/data-platform/token
- [ ] Register free OpenAQ account for higher rate limits
- [ ] Build `data_scraper.py` with all 4 adapters + normaliser
- [ ] Build `retrain_agent.py` with evaluation + R2 upload
- [ ] Set up Celery + Upstash Redis connection
- [ ] Write Alembic migrations for new tables
- [ ] Backfill: run historical Kaggle data into `aqi_readings` table as baseline

---

### Phase 3 — Database Schema & Backend API
**Goal**: Full FastAPI backend with all endpoints needed by frontend and alert engine.

**Database schema** (PostgreSQL / Supabase):
```sql
users               — id, clerk_user_id, email, name, phone_whatsapp, created_at
user_locations      — id, user_id, city, latitude, longitude, is_primary
subscriptions       — id, user_id, plan (free/pro), status, stripe_customer_id
alert_preferences   — id, user_id, channels (email/sms/whatsapp), aqi_threshold, advance_hours
alerts_sent         — id, user_id, city, aqi_level, message_en, message_ur, channel, sent_at, status
aqi_readings        — (see Phase 2)
forecasts           — id, city, forecast_for, pm25_predicted, lower_ci, upper_ci, generated_at
model_versions      — (see Phase 2)
```

**API endpoints** (`apps/api/main.py`):
```
POST /auth/webhook              — Clerk webhook: create user row on signup
GET  /users/me                  — Get current user profile + preferences
PUT  /users/me                  — Update profile (phone, locations, prefs)

GET  /aqi/current               — Live AQI for all cities (cached 5 min in Redis)
GET  /aqi/{city}/history        — Last 7 days hourly readings
GET  /aqi/{city}/forecast       — 24-hour Prophet forecast with CI bands

GET  /alerts                    — User's alert history (paginated)
POST /alerts/test               — Send a test alert to verify channels

GET  /admin/model-status        — Model version history + current MAE
POST /admin/retrain             — Manually trigger retraining (admin only)
```

**Tasks**:
- [ ] Scaffold FastAPI app with folder structure (routers, models, schemas, services, workers)
- [ ] Connect to Supabase PostgreSQL via SQLAlchemy async
- [ ] Implement Clerk JWT middleware for auth
- [ ] Build all endpoints listed above
- [ ] Add Redis caching layer for `/aqi/current` (5-min TTL)
- [ ] Write Alembic migrations for all tables
- [ ] Add WebSocket endpoint `/ws/aqi` for real-time updates to dashboard

---

### Phase 4 — User Management & Alert Preferences
**Goal**: Users can register, set locations, choose alert channels and thresholds.

**User flows**:
1. **Signup**: email + password via Clerk → webhook creates `users` row → onboarding wizard
2. **Onboarding wizard** (3 steps):
   - Step 1: Select primary city (Islamabad / Karachi / Lahore / Peshawar / Quetta)
   - Step 2: Alert channels — email (free), WhatsApp (verify number), SMS (Pro plan)
   - Step 3: Threshold — "Alert me when AQI is: Moderate / Unhealthy / Very Unhealthy / Hazardous"
3. **Profile page**: update name, cities, channels, thresholds, Urdu language preference
4. **WhatsApp verification**: user sends "JOIN smokalert" to Twilio/Meta sandbox number to opt in
5. **SMS verification**: OTP sent via Twilio to verify Pakistani mobile number

**Subscription tiers**:
| Tier | Price | Channels | Cities | Alerts/day |
|------|-------|----------|--------|------------|
| Free | $0 | Email only | 1 city | 3 |
| Pro | $3/month | Email + WhatsApp + SMS | 5 cities | Unlimited |

**Tasks**:
- [ ] Integrate Clerk SDK in Next.js app
- [ ] Build onboarding wizard (3-step form)
- [ ] Build profile settings page
- [ ] WhatsApp opt-in flow (sandbox number for dev, production Business API after Meta approval)
- [ ] SMS OTP verification flow via Twilio Verify
- [ ] Subscription gating logic (middleware check on Pro-only features)

---

### Phase 5 — Alert Engine
**Goal**: Automated hourly job that checks forecasts, identifies at-risk users, and delivers bilingual alerts.

**Alert trigger logic** (`packages/ml/alert_engine.py`):
```python
Every hour (Celery Beat):
  1. Pull latest 24h forecast for each city from /api/forecast/{city}
  2. For each forecast hour where predicted_aqi >= user_threshold:
     - Find all users subscribed to that city
     - Check: has this user already been alerted for this city in last 6 hours? (Redis dedup key)
     - If not: generate alert message (English + Urdu)
     - Enqueue delivery tasks: send_email(), send_whatsapp(), send_sms()
  3. Log each sent alert to alerts_sent table
  4. Update Redis dedup key (TTL = 6 hours)
```

**Alert message template** (bilingual):
```
English:
⚠️ SmogAlert PK — [City] Air Quality Warning
AQI is forecast to reach [LEVEL] ([VALUE]) in the next [N] hours.
[Health advisory based on source: vehicular/industrial/crop burning]
Take precautions: wear N95 mask, avoid outdoor activity.
Disable alerts: smokalert.pk/settings

Urdu:
⚠️ اسموگ الرٹ پاکستان — [شہر] فضائی آلودگی انتباہ
[Urdu translation of the above]
```

**Alert channels** (`services/alert_dispatcher.py`):
- **Email**: Resend SDK → React Email template with AQI gauge chart (inline PNG)
- **WhatsApp**: Meta Cloud API template message (pre-approved template required)
- **SMS**: Twilio Messages API, 160-char limit, Urdu transliteration for brevity

**Tasks**:
- [ ] Build `alert_engine.py` Celery task with dedup logic
- [ ] Build `alert_dispatcher.py` with Email/WhatsApp/SMS senders
- [ ] Design React Email template (AQI gauge + forecast mini-chart)
- [ ] Register WhatsApp Business API with Meta (takes 1-3 days for approval)
- [ ] Create and get approved WhatsApp message template with Meta
- [ ] Twilio account setup + Pakistan SMS verification
- [ ] Resend account + domain verification
- [ ] Write unit tests for alert trigger logic

---

### Phase 6 — Frontend Web App (Next.js)
**Goal**: Production-quality web app replacing the Streamlit dashboard.

**Pages & components**:

```
/                       — Landing page
  - Hero: "Get air quality alerts before it's too dangerous"
  - City AQI cards (live, auto-refresh 5 min)
  - How it works (3-step)
  - Pricing table (Free vs Pro)
  - Footer

/dashboard              — Main authenticated view
  - AQI map (React Leaflet, Pakistan cities with colour-coded markers)
  - City selector tabs
  - Current AQI gauge + pollutant breakdown (PM2.5, PM10, NO2, SO2, CO, O3)
  - 24-hour forecast chart (Prophet output with CI bands — Recharts)
  - Anomaly timeline (last 7 days)
  - Source classification panel (Industrial / Vehicular / Crop Burning)
  - Live feed badge (last updated: X minutes ago)

/alerts                 — Alert history
  - Table: city, timestamp, AQI level, channel, status (delivered/failed)
  - Filter by city, date range, channel

/settings               — User preferences
  - Profile info
  - Manage cities (add/remove up to plan limit)
  - Alert channels (email / WhatsApp / SMS) + verify buttons
  - AQI threshold slider
  - Notification frequency (immediate / digest)
  - Language preference (English / Urdu)
  - Test alert button

/onboarding             — Post-signup wizard (3 steps)

/admin                  — Admin-only dashboard
  - Model version table (MAE history per city)
  - Data ingestion status (last scrape timestamps per source)
  - Manual retrain trigger button
  - User count / alert delivery metrics
```

**UI/UX notes**:
- Mobile-first responsive design
- Pakistan-appropriate colour palette (green/teal primary)
- Urdu RTL text support via `dir="rtl"` on Urdu content blocks
- AQI colour scale: Green / Yellow / Orange / Red / Purple / Maroon (US EPA standard)
- Dark mode support via Tailwind dark variant

**Tasks**:
- [ ] Scaffold Next.js 14 app with Tailwind + shadcn/ui
- [ ] Set up Clerk provider + middleware for protected routes
- [ ] Build landing page
- [ ] Build dashboard page with map + charts
- [ ] Build alerts history page
- [ ] Build settings/onboarding pages
- [ ] Build admin panel
- [ ] Integrate Supabase Realtime for live AQI WebSocket updates
- [ ] Add PWA manifest (enables "Add to Home Screen" on mobile — bridge to mobile app)

---

### Phase 7 — ML Pipeline Integration into API
**Goal**: Serve live Prophet forecasts from the FastAPI backend, with models auto-loaded from Cloudflare R2.

**Changes to existing ML code**:
- `model.py`: add `--city` CLI flag to retrain only one city (faster incremental retraining)
- `model.py`: save per-city Prophet models as `prophet_{city}.pkl` (already exists in `models/`)
- New `packages/ml/forecast_service.py`: loads city model from R2, runs 24h forecast, returns JSON
- New `packages/ml/model_loader.py`: caches model in memory, invalidates when new version uploaded to R2

**Forecast API flow**:
```
GET /aqi/{city}/forecast
  → Check Redis cache (TTL 1h)
  → If miss: forecast_service.predict(city, horizon=24)
    → Load prophet_{city}.pkl from memory cache (or R2 if stale)
    → Run model.predict(future_df)
    → Return [{hour, pm25_predicted, lower_ci, upper_ci, aqi_category}]
  → Store result in Redis
  → Store result in forecasts table
```

**Tasks**:
- [ ] Build `forecast_service.py` and `model_loader.py`
- [ ] Wire up `/aqi/{city}/forecast` endpoint
- [ ] Set up Cloudflare R2 bucket + boto3/s3-compatible client
- [ ] Modify `retrain_agent.py` to upload new models to R2 after evaluation
- [ ] Benchmark forecast endpoint latency (target < 500ms with Redis cache)

---

### Phase 8 — Deployment

**Step-by-step deployment sequence**:

#### 8.1 Supabase Setup (Database)
1. Create project at supabase.com (free tier)
2. Run Alembic migrations against Supabase PostgreSQL URL
3. Enable Row Level Security on `users`, `alert_preferences`, `alerts_sent`
4. Enable Realtime on `aqi_readings` table

#### 8.2 Upstash Redis Setup
1. Create Redis database at upstash.com (free tier)
2. Copy `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` to env vars

#### 8.3 Cloudflare R2 Setup (Model Storage)
1. Create R2 bucket `smokalert-models` at cloudflare.com (free 10 GB)
2. Generate R2 API token → add to env vars

#### 8.4 Railway Setup (Backend + Workers)
1. Create Railway project
2. Service 1: FastAPI (`apps/api`) — expose port 8000
3. Service 2: Celery worker (`celery -A worker worker`)
4. Service 3: Celery Beat (`celery -A worker beat`)
5. Set all env vars in Railway dashboard
6. Connect Railway to GitHub repo for auto-deploy on `main` push

#### 8.5 Vercel Setup (Frontend)
1. Import GitHub repo into Vercel
2. Set root directory to `apps/web`
3. Add all env vars (NEXT_PUBLIC_API_URL, Clerk keys, Supabase URL)
4. Auto-deploys on every push to `main`

#### 8.6 Third-party Service Setup
1. **Clerk**: Create app → get publishable + secret keys
2. **Resend**: Verify sending domain → get API key
3. **Twilio**: Create account → get SID + auth token → buy Pakistan-capable number
4. **Meta WhatsApp Cloud API**: Create Meta Business account → create app → apply for WhatsApp Business API (1-3 days approval)
5. **WAQI**: Register free token at aqicn.org/data-platform/token
6. **OpenAQ**: Register at api.openaq.org for higher rate limits

#### 8.7 GitHub Actions CI/CD
```yaml
# .github/workflows/deploy.yml
# On push to main:
#   - Run Python tests (pytest)
#   - Run Next.js type check (tsc)
#   - Railway auto-deploys API
#   - Vercel auto-deploys frontend
#   - Run scheduled retrain_agent.py via Actions cron (daily 2 AM PKT)
```

**Environment variables required**:
```
# Database
SUPABASE_URL
SUPABASE_SERVICE_KEY
DATABASE_URL

# Redis
UPSTASH_REDIS_REST_URL
UPSTASH_REDIS_REST_TOKEN

# Auth
CLERK_SECRET_KEY
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY

# Alerts
RESEND_API_KEY
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_PHONE_NUMBER
META_WHATSAPP_TOKEN
META_WHATSAPP_PHONE_ID

# AQI APIs
WAQI_TOKEN
OPENAQ_API_KEY

# Model Storage
CLOUDFLARE_R2_ACCESS_KEY
CLOUDFLARE_R2_SECRET_KEY
CLOUDFLARE_R2_BUCKET
CLOUDFLARE_R2_ENDPOINT

# App
NEXT_PUBLIC_API_URL
ADMIN_USER_IDS  # comma-separated Clerk user IDs for admin access
```

---

### Phase 9 — Mobile App (Future Phase)

**After web app is stable** (~3-4 months after Phase 8):

**Tech**: React Native + Expo (maximum code reuse from Next.js)

**Features**:
- Native AQI map (React Native Maps)
- Push notifications via Expo Push + Firebase FCM
- Background location for local AQI readings
- Offline-first: cache last 24h forecast in AsyncStorage
- iOS App Store + Google Play Store release

**Timeline**: Start after 200+ active web users for validation.

---

## Execution Sequence

| Phase | Estimated Effort | Depends On |
|-------|-----------------|------------|
| Phase 1 — Repo Restructure | 1 day | — |
| Phase 2 — Data Agent + Scraper | 2-3 days | Phase 1 |
| Phase 3 — Database + FastAPI | 3-4 days | Phase 2 |
| Phase 4 — User Management | 2 days | Phase 3 |
| Phase 5 — Alert Engine | 2-3 days | Phases 3 + 4 |
| Phase 6 — Frontend | 5-7 days | Phase 3 |
| Phase 7 — ML Integration | 2 days | Phases 2 + 3 |
| Phase 8 — Deployment | 1-2 days | All above |
| **Total** | **~18-22 days** | |

---

## Key Decisions & Trade-offs

1. **FastAPI over Django**: keeps Python ML in same process, simpler, lighter. Django only needed if heavy admin features are required later.

2. **Clerk over Supabase Auth**: better DX, pre-built UI components, easier phone/WhatsApp number collection. Supabase Auth is still used for DB RLS.

3. **Celery over AWS Lambda for workers**: Railway free tier avoids cold starts; Celery Beat for scheduling is simpler than EventBridge. Switch to Lambda only if scale demands serverless.

4. **Rule-based source classifier kept as-is**: interpretable, no GPU needed, good enough for alert context. Upgrade to ML classifier later if accuracy data available.

5. **Prophet per-city models** (not one global model): city-specific seasonality patterns differ significantly (Lahore smog season ≠ Karachi sea breeze patterns).

6. **WhatsApp over SMS as primary channel**: 93% WhatsApp penetration in Pakistan; cheaper per message; richer templates. SMS is fallback/Pro-only.

7. **Monorepo**: easier to share types between `apps/api` and `apps/web`, single CI pipeline, simpler for solo/small team.

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Meta WhatsApp API approval delayed | Medium | Use Twilio WhatsApp sandbox in dev; plan 2-week buffer |
| OpenAQ missing cities (Peshawar, Quetta) | High | WAQI covers all 5; OpenMeteo as fallback |
| Railway free credit exhausted | Medium | Optimize worker sleep intervals; upgrade to $20/mo if needed |
| Prophet retrain too slow on free CPU | Medium | Retrain only changed cities; use pre-built models from R2 |
| Supabase 500 MB limit hit | Low | Archive old `aqi_readings` rows >6 months; compress |
| Pakistan ISP blocking Twilio SMS | Low | Use local resellers (Jazz/Telenor SMS gateway API) as backup |
