"""
Alert dispatcher — sends alerts via email (Resend), SMS (Twilio), and
WhatsApp (Meta Cloud API).  Each channel is a no-op if the relevant API
key is not configured, so the system degrades gracefully during local dev.
"""

import httpx
import resend

from core.config import settings
from models.user import AlertSent, User


AQI_HEALTH_ADVICE = {
    "Moderate": "Unusually sensitive people should reduce prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Sensitive groups (elderly, children, heart/lung conditions) should reduce outdoor activity.",
    "Unhealthy": "Everyone may begin to experience health effects. Wear an N95 mask outdoors.",
    "Very Unhealthy": "Health alert: everyone should avoid prolonged outdoor exertion. N95 mask essential.",
    "Hazardous": "Health emergency! Stay indoors, seal windows. Avoid all outdoor activity.",
    "Test": "This is a test. Your SmogAlert PK notifications are working correctly.",
}

# AQI level → hex colour used in email template
_AQI_COLOUR = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
    "Test": "#0ea5e9",
}


# ── Email (Resend) ────────────────────────────────────────────────────────────

def _build_email_html(city: str, aqi_level: str, aqi_value: float | None, message_en: str) -> str:
    """Inline-HTML email body — works in all major email clients."""
    advice = AQI_HEALTH_ADVICE.get(aqi_level, "")
    aqi_display = f"{aqi_value:.0f}" if aqi_value is not None else "N/A"
    colour = _AQI_COLOUR.get(aqi_level, "#666")

    return f"""
    <div style="font-family:system-ui,sans-serif;max-width:600px;margin:auto;padding:24px;">
      <h2 style="color:{colour};margin-bottom:4px;">⚠️ SmogAlert PK — {city}</h2>
      <p style="font-size:20px;font-weight:bold;color:{colour};margin:0 0 12px;">
        {aqi_level} ({aqi_display} µg/m³ PM2.5)
      </p>
      <p style="color:#374151;">{message_en}</p>
      <div style="background:#f3f4f6;border-radius:8px;padding:12px;margin:16px 0;color:#374151;">
        {advice}
      </div>
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0;"/>
      <p style="font-size:12px;color:#9ca3af;">
        Manage your alerts at
        <a href="https://smokalert.pk/settings" style="color:#0d9488;">smokalert.pk/settings</a>
      </p>
    </div>
    """


async def _send_email(email: str, city: str, aqi_level: str, aqi_value: float | None, message_en: str) -> str:
    """Send an HTML alert email via Resend. Returns delivery status string."""
    if not settings.resend_api_key:
        print("[alert_dispatcher] RESEND_API_KEY not set — skipping email")
        return "skipped_no_config"

    resend.api_key = settings.resend_api_key
    html_body = _build_email_html(city, aqi_level, aqi_value, message_en)

    resend.Emails.send({
        "from": settings.from_email,
        "to": [email],
        "subject": f"⚠️ SmogAlert PK — {city} Air Quality {aqi_level}",
        "html": html_body,
    })
    return "delivered"


# ── SMS (Twilio) ──────────────────────────────────────────────────────────────

async def _send_sms(phone: str, message: str) -> str:
    """Send an SMS via Twilio Messages API. Returns delivery status string."""
    if not (settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_phone_number):
        print("[alert_dispatcher] Twilio not configured — skipping SMS")
        return "skipped_no_config"

    if not phone:
        return "skipped_no_phone"

    # Twilio REST API — basic auth, no SDK dependency needed
    url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Messages.json"
    # Truncate to 160 chars for a single SMS segment
    sms_body = message[:157] + "..." if len(message) > 160 else message

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            auth=(settings.twilio_account_sid, settings.twilio_auth_token),
            data={"From": settings.twilio_phone_number, "To": phone, "Body": sms_body},
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Twilio error {resp.status_code}: {resp.text}")

    return "delivered"


# ── WhatsApp (Meta Cloud API) ─────────────────────────────────────────────────

async def _send_whatsapp(phone: str, city: str, aqi_level: str, message_en: str) -> str:
    """
    Send a WhatsApp message via the Meta Cloud API.

    Uses a free-form text message (works in the 24-hour customer service window).
    For production use you should create an approved template message in the Meta
    Business dashboard and call the template endpoint instead.

    Returns delivery status string.
    """
    if not (settings.meta_whatsapp_token and settings.meta_whatsapp_phone_id):
        print("[alert_dispatcher] Meta WhatsApp API not configured — skipping WhatsApp")
        return "skipped_no_config"

    if not phone:
        return "skipped_no_phone"

    # Normalise phone to E.164 without leading +
    clean_phone = phone.lstrip("+")

    url = f"https://graph.facebook.com/v19.0/{settings.meta_whatsapp_phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.meta_whatsapp_token}",
        "Content-Type": "application/json",
    }
    body = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": clean_phone,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": f"⚠️ *SmogAlert PK — {city}*\n{aqi_level}\n\n{message_en}\n\nManage alerts: smokalert.pk/settings",
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Meta WhatsApp error {resp.status_code}: {resp.text}")

    return "delivered"


# ── Dispatcher ────────────────────────────────────────────────────────────────

async def dispatch_alert(
    user: User,
    city: str,
    aqi_level: str,
    aqi_value: float | None,
    message_en: str,
    message_ur: str,
    channels: str = "email",
) -> None:
    """
    Send an alert to the user via all their configured channels and log each
    attempt to the alerts_sent table.

    Uses its own DB session so the caller's session is never contaminated by
    a mid-dispatch commit.
    """
    from core.database import AsyncSessionLocal

    channel_list = [c.strip() for c in channels.split(",")]

    rows: list[AlertSent] = []
    for channel in channel_list:
        status = "pending"
        try:
            if channel == "email":
                status = await _send_email(user.email, city, aqi_level, aqi_value, message_en)

            elif channel == "sms":
                status = await _send_sms(user.phone_whatsapp or "", message_ur)

            elif channel == "whatsapp":
                status = await _send_whatsapp(
                    user.phone_whatsapp or "", city, aqi_level, message_en
                )

            else:
                status = "skipped_unknown_channel"

        except Exception as exc:
            print(f"[alert_dispatcher] {channel} send failed for user {user.id}: {exc}")
            status = "failed"

        rows.append(AlertSent(
            user_id=user.id,
            city=city,
            aqi_level=aqi_level,
            aqi_value=aqi_value,
            message_en=message_en,
            message_ur=message_ur,
            channel=channel,
            status=status,
        ))

    async with AsyncSessionLocal() as db:
        db.add_all(rows)
        await db.commit()
