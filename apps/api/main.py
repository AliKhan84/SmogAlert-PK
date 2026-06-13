"""SmogAlert PK — FastAPI backend entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import engine, Base
from routers import admin, alerts, aqi, auth, users

app = FastAPI(
    title="SmogAlert PK API",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://smokalert.pk",
        "https://smokalert.vercel.app",
        "https://web-chi-lac-66.vercel.app",
        "https://web-dv2snmk8u-alikhan84s-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(aqi.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(admin.router, prefix="/api")


@app.on_event("startup")
async def create_tables():
    """Create all tables on startup if they don't exist (dev convenience).
    Non-fatal: if DB is unreachable at boot, log and continue — app still serves /health.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        print(f"[startup] DB connection failed — tables not created: {exc}")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "SmogAlert PK API"}
