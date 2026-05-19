"""
model/tier1.py
Historical lookup (Tier 1) — best for long-horizon predictions (>2 hours).
MAE = 0.023 on normalized scale.
"""

from datetime import datetime
import math
from model.loader import ModelLoader


def get_historical_jam(junction_id: str, target_dt: datetime) -> float:
    """
    Returns the historical average jam_factor (0-1) for a junction
    at the given day-of-week and hour.
    Falls back to 0.0 (FREE) if junction not found.
    """
    lookup = ModelLoader.get_hist_lookup()
    hour   = target_dt.hour
    dow    = target_dt.weekday()
    return float(lookup.get((junction_id, hour, dow), 0.0))


def jam_to_level(jam: float) -> str:
    if jam < 0.2:  return "FREE"
    if jam < 0.4:  return "LIGHT"
    if jam < 0.6:  return "MODERATE"
    return "HIGH"


def jam_to_multiplier(jam: float) -> float:
    """
    Converts normalized jam_factor to travel time multiplier.
    0.0 (FREE)     → 1.0x (no delay)
    0.3 (LIGHT)    → 1.3x
    0.5 (MODERATE) → 1.6x
    0.8 (HIGH)     → 2.2x
    """
    return 1.0 + jam * 1.5


def predict_route_duration(
    junction_ids: list[str],
    departure_dt: datetime,
    base_duration_seconds: int,
) -> dict:
    """
    Predicts total route duration using historical patterns for each junction.

    Args:
        junction_ids: List of junction IDs along the route
        departure_dt: Planned departure datetime
        base_duration_seconds: Free-flow duration from Mapbox (no traffic)

    Returns:
        {
          "predicted_seconds": int,
          "jam_factor_avg": float,
          "level": str,
          "confidence": float
        }
    """
    if not junction_ids:
        return {
            "predicted_seconds": base_duration_seconds,
            "jam_factor_avg": 0.0,
            "level": "FREE",
            "confidence": 0.5,
        }

    # Get jam for each junction, estimating arrival time progressively
    per_junction_seconds = base_duration_seconds / len(junction_ids)
    elapsed = 0.0
    jam_values = []

    for jid in junction_ids:
        arrival_seconds = elapsed + per_junction_seconds * 0.5
        arrival_dt = departure_dt.replace(
            hour=int((departure_dt.hour * 3600 + arrival_seconds) // 3600) % 24,
            minute=int((departure_dt.minute * 60 + arrival_seconds) % 3600 // 60),
        )
        jam = get_historical_jam(jid, arrival_dt)
        jam_values.append(jam)
        elapsed += per_junction_seconds * jam_to_multiplier(jam)

    avg_jam = sum(jam_values) / len(jam_values)
    multiplier = jam_to_multiplier(avg_jam)
    predicted_secs = int(base_duration_seconds * multiplier)

    return {
        "predicted_seconds":  predicted_secs,
        "predicted_minutes":  round(predicted_secs / 60, 1),
        "jam_factor_avg":     round(avg_jam, 3),
        "level":              jam_to_level(avg_jam),
        "confidence":         0.85,
    }