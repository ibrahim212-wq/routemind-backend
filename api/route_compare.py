"""
api/route_compare.py
POST /api/route-compare

Compares multiple routes and ranks them by predicted congestion.
Used by Route AI Pick feature.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
import logging

from model.tier1 import predict_route_duration
from services.junction_mapper import map_route_to_junctions

logger = logging.getLogger("routemind.route_compare")
router = APIRouter()


class Waypoint(BaseModel):
    lat: float
    lng: float


class RouteOption(BaseModel):
    route_id:              str
    waypoints:             list[Waypoint]
    base_duration_seconds: int


class RouteCompareRequest(BaseModel):
    routes:       list[RouteOption]
    departure_dt: str   # ISO datetime


class RouteResult(BaseModel):
    route_id:           str
    rank:               int
    predicted_minutes:  float
    jam_factor_avg:     float
    level:              str
    delay_minutes:      float
    is_recommended:     bool


class RouteCompareResponse(BaseModel):
    results:          list[RouteResult]
    recommended_id:   str
    departure_time:   str


@router.post("/route-compare", response_model=RouteCompareResponse)
async def route_compare(req: RouteCompareRequest):
    departure_dt = datetime.fromisoformat(req.departure_dt)
    results = []

    for route in req.routes:
        waypoints = [{"lat": w.lat, "lng": w.lng} for w in route.waypoints]
        junctions = map_route_to_junctions(waypoints)
        junction_ids = [j["junction_id"] for j in junctions]

        pred = predict_route_duration(
            junction_ids,
            departure_dt,
            route.base_duration_seconds,
        )
        free_min = route.base_duration_seconds / 60
        delay    = max(0.0, pred["predicted_minutes"] - free_min)

        results.append({
            "route_id":          route.route_id,
            "predicted_minutes": pred["predicted_minutes"],
            "jam_factor_avg":    pred["jam_factor_avg"],
            "level":             pred["level"],
            "delay_minutes":     round(delay, 1),
        })

    # Rank by predicted time
    results.sort(key=lambda r: r["predicted_minutes"])
    best_id = results[0]["route_id"]

    route_results = [
        RouteResult(
            route_id          = r["route_id"],
            rank              = i + 1,
            predicted_minutes = r["predicted_minutes"],
            jam_factor_avg    = r["jam_factor_avg"],
            level             = r["level"],
            delay_minutes     = r["delay_minutes"],
            is_recommended    = r["route_id"] == best_id,
        )
        for i, r in enumerate(results)
    ]

    return RouteCompareResponse(
        results         = route_results,
        recommended_id  = best_id,
        departure_time  = departure_dt.strftime("%H:%M"),
    )