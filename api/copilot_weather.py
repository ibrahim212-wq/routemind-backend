"""
api/copilot_weather.py — the `weather` tool: the sky where the driver is, or where they are going.

What a co-driver in Egypt needs from weather is not a forecast page, it is the few things that change a drive:
fog (الشبورة) and dust (الخماسين) that take visibility away, rain on roads that flood, gusts on desert roads,
heat. Open-Meteo gives current conditions + the next hours for a point with no key; one call per question,
no cost per request on the free tier.

Licensing: the keyless endpoint is Open-Meteo's non-commercial tier. For a commercial launch set
OPEN_METEO_API_KEY — the same request then goes to the customer endpoint (a subscription, not per call).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("routemind.copilot.weather")

_KEY = os.getenv("OPEN_METEO_API_KEY", "")
URL = ("https://customer-api.open-meteo.com/v1/forecast" if _KEY
       else "https://api.open-meteo.com/v1/forecast")
_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0))

# WMO weather interpretation codes → one plain English word the model re-says in the turn's language.
_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "heavy rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def describe(code: Optional[int]) -> str:
    try:
        return _WMO.get(int(code), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def summarize(body: Dict[str, Any], where: str) -> Dict[str, Any]:
    """Pure: the provider's JSON → the tool result the model speaks from (plus driving flags)."""
    cur = body.get("current") or {}
    hourly = body.get("hourly") or {}
    out: Dict[str, Any] = {"available": True, "where": where, "conditions": describe(cur.get("weather_code"))}
    for src, dst in (("temperature_2m", "temp_c"), ("apparent_temperature", "feels_like_c"),
                     ("wind_speed_10m", "wind_kmh"), ("wind_gusts_10m", "gusts_kmh")):
        v = cur.get(src)
        if v is not None:
            out[dst] = round(float(v))
    vis = cur.get("visibility")
    if vis is not None:
        out["visibility_km"] = round(float(vis) / 1000.0, 1)
    probs = [p for p in (hourly.get("precipitation_probability") or [])[:3] if p is not None]
    if probs:
        out["rain_chance_next_3h_pct"] = int(max(probs))
    later = [describe(c) for c in (hourly.get("weather_code") or [])[:3]]
    if later and any(c != out["conditions"] for c in later):
        out["next_hours"] = later
    flags = []
    if out.get("visibility_km") is not None and out["visibility_km"] < 1.0:
        flags.append("LOW VISIBILITY — tell them to slow down, low beams, long gaps")
    if out["conditions"] in ("fog", "freezing fog"):
        flags.append("fog (الشبورة)")
    if out.get("gusts_kmh", 0) >= 50:
        flags.append("strong gusts — dust / crosswind on open roads")
    if "rain" in out["conditions"] or "thunder" in out["conditions"] or out.get("rain_chance_next_3h_pct", 0) >= 60:
        flags.append("rain — slippery roads, flooded underpasses")
    if out.get("temp_c", 0) >= 40:
        flags.append("extreme heat")
    if flags:
        out["driving_flags"] = flags
    out["note"] = ("Say it in one or two short spoken sentences: the conditions, the temperature rounded, and "
                   "ONLY the driving flags that exist. Never read the numbers as a list.")
    return out


async def weather(ctx: Dict[str, Any], where: str = "here") -> Dict[str, Any]:
    where = "destination" if where == "destination" else "here"
    if where == "destination":
        lat, lng = ctx.get("dest_lat"), ctx.get("dest_lng")
    else:
        lat, lng = ctx.get("user_lat"), ctx.get("user_lng")
    if lat is None or lng is None:
        return {"available": False,
                "note": f"No {'destination' if where == 'destination' else 'GPS'} position to look up — say so."}
    params = {"latitude": f"{float(lat):.3f}", "longitude": f"{float(lng):.3f}",
              "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,wind_gusts_10m,visibility",
              "hourly": "weather_code,precipitation_probability", "forecast_hours": "3",
              "timezone": "Africa/Cairo", "wind_speed_unit": "kmh"}
    if _KEY:
        params["apikey"] = _KEY
    try:
        resp = await _client.get(URL, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        return summarize(resp.json(), where)
    except Exception as e:
        logger.warning(f"copilot weather failed: {e}")
        return {"available": False, "note": "The weather service didn't answer — say you can't check it right now."}
