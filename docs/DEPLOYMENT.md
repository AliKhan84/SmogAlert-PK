# SmogAlert PK — Deployment Guide (Phases 1–8)

All services run on **free tiers**. Follow the steps in order.

---

## Step 1 — Register Free Accounts (do this first, in parallel)

| Service | URL | What to get |
|---------|-----|-------------|
| Supabase | supabase.com | Project URL + service role key + DB connection string |
| Upstash | upstash.com | Redis REST URL + token |
| Clerk | clerk.com | Publishable key + secret key |
| Resend | resend.com | API key (verify your domain or use their shared domain for testing) |
| Railway | railway.app | Account (no keys needed yet) |
| Vercel | vercel.com | Account (no keys needed yet) |
| WAQI | aqicn.org/data-platform/token | Free API token (instant) |

---

## Step 2 — Supabase Database Setup

1. Create a new project at supabase.com
2. Go to Project Settings → Database → copy `Connection String (URI)`
3. Replace `[YOUR-PASSWORD]` with your DB password
4. Enable Realtime on the `aqi_readings` table (Database → Replication)
5. Run migrations:
   ```bash
   cd apps/api
   DATABASE_URL="postgresql+psycopg2://<your-supabase-url>" alembic upgrade head
   ```

---

## Step 3 — Configure Environment Variables

```bash
cp .env.example .env
cp apps/web/.env.local.example apps/web/.env.local
# Fill in all values from Step 1
```

---

## Step 4 — Backfill Historical Data

Load existing Kaggle data into the `aqi_readings` table so the dashboard isn't empty on launch:

```bash
python agents/backfill_kaggle_data.py
```

---

## Step 5 — Run Locally (verify everything works)

```bash
# Terminal 1 — API
cd apps/api && pip install -r requirements.txt && uvicorn main:app --reload

# Terminal 2 — Celery worker (optional for local dev)
cd apps/api && celery -A workers.celery_app worker --loglevel=info

# Terminal 3 — Frontend
cd apps/web && npm run dev
```

Open http://localhost:3000 — you should see the landing page with AQI data.

---

## Step 6 — Deploy Backend to Railway

1. Go to railway.app → New Project → Deploy from GitHub repo
2. Select the `SmogAlert-PK` repository
3. Set root directory: `apps/api`
4. Railway auto-detects Python + runs `uvicorn main:app`
5. Add all environment variables from `.env` in Railway dashboard
6. Create **2 additional services** in the same Railway project:
   - **Celery Worker**: start command = `celery -A workers.celery_app worker --loglevel=info --concurrency=2`
   - **Celery Beat**: start command = `celery -A workers.celery_app beat --loglevel=info`
7. Note the API URL (e.g., `https://smokalert-api.up.railway.app`)

---

## Step 7 — Deploy Frontend to Vercel

1. Go to vercel.com → New Project → Import from GitHub
2. Set root directory: `apps/web`
3. Add environment variables:
   - `NEXT_PUBLIC_API_URL` = your Railway API URL + `/api`
   - All Clerk keys
4. Deploy → copy the Vercel URL

---

## Step 8 — Configure Clerk Webhook

1. In Clerk dashboard → Webhooks → Add endpoint
2. URL: `https://your-railway-api.railway.app/api/auth/webhook`
3. Events to subscribe: `user.created`, `user.deleted`
4. Copy the signing secret → add as `CLERK_WEBHOOK_SECRET` in Railway env vars

---

## Step 9 — Add GitHub Actions Secrets

In your GitHub repo → Settings → Secrets → Actions:
- `DATABASE_URL`
- `WAQI_TOKEN`
- `OPENAQ_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

This enables the hourly scrape cron job as a fallback when Railway sleeps.

---

## Step 10 — Get WAQI Token

1. Visit aqicn.org/data-platform/token
2. Enter your email → receive token in 1-2 minutes
3. Add as `WAQI_TOKEN` in Railway and GitHub Secrets

---

## Verify Deployment Checklist

- [ ] Landing page loads at Vercel URL
- [ ] AQI gauges show data (calls `/api/aqi/current`)
- [ ] Sign up works and redirects to `/onboarding`
- [ ] Dashboard loads with map and forecast chart
- [ ] Settings page saves preferences
- [ ] Test alert sends email to your address
- [ ] Admin panel accessible at `/admin` for your Clerk user ID

---

## Monthly Cost

| Service | Cost |
|---------|------|
| Vercel | $0 |
| Railway (3 services) | ~$0 (within $5 free credit) |
| Supabase | $0 (under 500 MB) |
| Upstash Redis | $0 (under 10K commands/day) |
| Cloudflare R2 | $0 (under 10 GB) |
| Resend | $0 (under 3K emails/month) |
| WAQI API | $0 |
| **Total** | **$0/month** |

Domain: use `yourapp.vercel.app` for free, or buy `.pk` domain for ~$10/year.
