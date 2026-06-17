"""
RouteMind AI Backend — FastAPI
==============================
Deploy on Google Cloud Run.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from services.scheduler import start_scheduler
import json
import os

from model.loader import ModelLoader
from api.plan_drive import router as plan_drive_router
from api.scan_route import router as scan_route_router
from api.route_compare import router as route_compare_router
from api.plan_drive_stream import router as plan_drive_stream_router
from api.trip_alert import router as trip_alert_router          # ← جديد

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("routemind")


def init_firebase():
    try:
        import firebase_admin
        from firebase_admin import credentials

        # Service-account Certificate credential from Secret Manager. This is the
        # battle-tested firebase-admin path (key-based JWT bearer exchange), which
        # — unlike metadata-server ADC — reliably obtains the firebase.messaging
        # scope required for FCM v1 delivery.
        creds_json = os.environ.get("FIREBASE_CREDENTIALS")
        if creds_json:
            cred = credentials.Certificate(json.loads(creds_json))
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized from Secret Manager ✓")
            return

        # Explicit key file fallback (only when GOOGLE_APPLICATION_CREDENTIALS is
        # set) — no implicit default path so a stray bundled key can't load.
        key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if key_path and os.path.exists(key_path):
            firebase_admin.initialize_app(credentials.Certificate(key_path))
            logger.info("Firebase initialized from key file ✓")
            return

        logger.warning("Firebase credentials not found — notifications disabled")

    except Exception as e:
        logger.warning(f"Firebase init failed: {e} — notifications disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RouteMind AI model...")
    ModelLoader.load()
    logger.info("Model ready ✓")
    init_firebase()
    start_scheduler()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="RouteMind AI Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(plan_drive_router,        prefix="/api")
app.include_router(scan_route_router,        prefix="/api")
app.include_router(route_compare_router,     prefix="/api")
app.include_router(plan_drive_stream_router, prefix="/api")
app.include_router(trip_alert_router,        prefix="/api")     # ← جديد


@app.post("/api/test-notify")
async def test_notify(body: dict):
    """
    Test endpoint — يبعت notification فورية لـ user معين.
    للتأكد إن الـ FCM pipeline شغال.
    """
    try:
        from firebase_admin import messaging
        from services.supabase_client import get_supabase

        user_id = body.get("user_id")
        if not user_id:
            return {"status": "error", "reason": "user_id مطلوب"}

        supabase = get_supabase()
        if not supabase:
            return {"status": "error", "reason": "Supabase مش متصل"}

        res   = supabase.table("fcm_tokens").select("token").eq("user_id", user_id).limit(1).execute()
        token = res.data[0]["token"] if res.data else None

        if not token:
            return {"status": "error", "reason": f"مفيش FCM token للـ user {user_id}"}

        message_name = messaging.send(messaging.Message(
            notification=messaging.Notification(
                title="RouteMind Test 🚦",
                body="الـ backend شغال وبيبعت notifications ✅",
            ),
            data={
                "type":       "test",
                "tap_action": "show_route_details",
            },
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="routemind_trips",
                    click_action="FLUTTER_NOTIFICATION_CLICK",
                ),
            ),
            token=token,
        ))

        return {
            "status":        "sent",
            "user_id":       user_id,
            "message_name":  message_name,
            "token_preview": token[:20] + "...",
        }

    except Exception as e:
        return {"status": "error", "reason": str(e)}


@app.post("/api/run-scan")
async def run_scan_now():
    """
    بيشغّل الـ intelligent scan فوراً (بدل انتظار الـ scheduler).
    للاختبار + ممكن يتربط بـ Cloud Scheduler كضمان إضافي.
    """
    try:
        from services.scanner import run_intelligent_scan
        from services.supabase_client import get_supabase

        supabase = get_supabase()
        if not supabase:
            return {"status": "error", "reason": "Supabase مش متصل"}

        await run_intelligent_scan(supabase)
        return {"status": "scan_completed"}

    except Exception as e:
        logger.error(f"Manual scan failed: {e}")
        return {"status": "error", "reason": str(e)}


@app.post("/api/scan-now/{trip_id}")
async def scan_now(trip_id: str):
    """
    Immediately scan ONE trip — the app should call this right after creating a
    trip so short-notice trips (leaving soon) don't wait for the next scheduler
    cycle. Runs the same scan_trip path as the scheduler.
    """
    try:
        from services.scanner import scan_trip
        from services.supabase_client import get_supabase

        supabase = get_supabase()
        if not supabase:
            return {"status": "error", "reason": "Supabase مش متصل"}

        res  = supabase.table("planned_trips").select("*").eq("id", trip_id).single().execute()
        trip = res.data
        if not trip:
            return {"status": "error", "reason": f"trip {trip_id} not found"}

        # reverse_adj is unused by the route-level scan; pass an empty dict.
        await scan_trip(trip, {}, supabase)
        return {"status": "scanned", "trip_id": trip_id}

    except Exception as e:
        logger.error(f"scan-now failed for {trip_id}: {e}")
        return {"status": "error", "reason": str(e)}


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": ModelLoader.is_loaded(),
        "version":      "1.0.0",
    }