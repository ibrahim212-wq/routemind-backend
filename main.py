"""
RouteMind AI Backend — FastAPI
==============================
Deploy on Google Cloud Run.

Endpoints:
  POST /api/plan-drive     → 76 time slots with predictions (Tier 1 + Tier 2)
  POST /api/scan-route     → Smart route scan for notifications
  POST /api/route-compare  → Compare multiple routes
  GET  /health             → Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from model.loader import ModelLoader
from api.plan_drive import router as plan_drive_router
from api.scan_route import router as scan_route_router
from api.route_compare import router as route_compare_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("routemind")

# ── Startup: load model once into memory ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RouteMind AI model...")
    ModelLoader.load()
    logger.info("Model ready ✓")
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

app.include_router(plan_drive_router,    prefix="/api")
app.include_router(scan_route_router,    prefix="/api")
app.include_router(route_compare_router, prefix="/api")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": ModelLoader.is_loaded(),
        "version": "1.0.0",
    }