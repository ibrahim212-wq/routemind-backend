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

        creds_json = os.environ.get("FIREBASE_CREDENTIALS")
        if creds_json:
            creds_dict = json.loads(creds_json)
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized from Secret Manager ✓")
            return

        local_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "./firebase-service-account.json"
        )
        if os.path.exists(local_path):
            firebase_admin.initialize_app(credentials.Certificate(local_path))
            logger.info("Firebase initialized from local file ✓")
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

        messaging.send(messaging.Message(
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
            "token_preview": token[:20] + "...",
        }

    except Exception as e:
        return {"status": "error", "reason": str(e)}


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": ModelLoader.is_loaded(),
        "version":      "1.0.0",
    }