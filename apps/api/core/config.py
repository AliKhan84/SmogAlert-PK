"""Application configuration loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings

# Repo root is 2 levels above this file (apps/api/core/config.py → repo root)
_ROOT_ENV = Path(__file__).parents[3] / ".env"


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/smokalert"
    supabase_url: str = ""
    supabase_service_key: str = ""

    # Redis (Upstash)
    redis_url: str = "redis://localhost:6379/0"
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""

    # Auth (Clerk)
    clerk_secret_key: str = ""
    clerk_webhook_secret: str = ""

    # Email (Resend)
    resend_api_key: str = ""
    from_email: str = "alerts@smokalert.pk"

    # SMS (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # WhatsApp (Meta Cloud API)
    meta_whatsapp_token: str = ""
    meta_whatsapp_phone_id: str = ""

    # AQI Data APIs
    waqi_token: str = ""
    openaq_api_key: str = ""

    # Model Storage (Cloudflare R2)
    cloudflare_r2_access_key: str = ""
    cloudflare_r2_secret_key: str = ""
    cloudflare_r2_bucket: str = "smokalert-models"
    cloudflare_r2_endpoint: str = ""

    # App
    app_name: str = "SmogAlert PK"
    api_prefix: str = "/api"
    debug: bool = False
    admin_user_ids: str = ""  # comma-separated Clerk user IDs

    @property
    def admin_ids(self) -> list[str]:
        return [uid.strip() for uid in self.admin_user_ids.split(",") if uid.strip()]

    class Config:
        env_file = str(_ROOT_ENV)
        case_sensitive = False
        extra = "ignore"  # root .env has NEXT_PUBLIC_* vars the API doesn't need


settings = Settings()
