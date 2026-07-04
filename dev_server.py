"""
dev_server.py — LOCAL DEV ONLY, not deployed.

Minimal FastAPI app for exercising the 4 new "add stop along the route"
endpoints in api/places.py without booting the full backend (main.py also
loads the torch/torch-geometric ML model, APScheduler, and Firebase — none of
which those 4 endpoints touch).

Only needs: fastapi, uvicorn, pydantic, httpx (all already in requirements.txt).
Does NOT import main.py and is NOT wired into the Dockerfile.

Run:
    uvicorn dev_server:app --host 0.0.0.0 --port 8080

Env (export before running — same vars main.py uses):
    GOOGLE_MAPS_API_KEY=...
    MAPBOX_TOKEN=...
"""

import logging

from fastapi import FastAPI

from api.places import router as places_router

# main.py calls this at import time; dev_server.py never imports main.py (by
# design), so without this our logger.info/.debug/.warning calls in api/places.py
# (including the [GOOGLE DEBUG] / [MOJIBAKE DEBUG] diagnostics) would mostly be
# silently dropped — Python's root logger defaults to WARNING with no handler.
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RouteMind dev_server (places only)")
app.include_router(places_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "dev_server (places only)"}
