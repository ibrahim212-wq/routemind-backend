"""
api/usage_limits.py — per-user daily limits (cost Phase 4, plan lever L; docs/cost/phase4-design.md §6 in the app
repo).

  copilot turns that reach the model   ≤ COPILOT_TURNS_PER_DAY (default 40) per user per Cairo day
  Plan-a-Drive plans (plan-drive-stream) ≤ PLANS_PER_DAY        (default 10) per user per Cairo day

The user is the app's device id (the `X-RouteMind-Device` header — the same UUID the app registers with its FCM
token); an old client without it is keyed by its address (the first X-Forwarded-For hop), which is weaker but
never unlimited. Deterministic fast-path turns (mute, repeat, volume, thanks, facts) cost nothing and are NOT
counted — only a turn that calls the model is.

The counters live in this instance's memory (like tts_clips' daily cap): with N Cloud Run instances a user can
reach N × the limit, and a restart resets it. A shared store (Firestore / Redis) or App Check is the stronger
form — listed in the report's Needs Khaled. Limits are read at call time so tests (and an env flip) change them.
"""
from __future__ import annotations

import os
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Mapping, Optional

try:  # Egypt observes DST again since 2023; tzdata may be missing on a slim image
    from zoneinfo import ZoneInfo
    _CAIRO = ZoneInfo("Africa/Cairo")
except Exception:  # pragma: no cover - fallback: EET without DST
    _CAIRO = timezone(timedelta(hours=2))

DEVICE_HEADER = "x-routemind-device"


def copilot_turns_per_day() -> int:
    return int(os.getenv("COPILOT_TURNS_PER_DAY", "40"))


def plans_per_day() -> int:
    return int(os.getenv("PLANS_PER_DAY", "10"))


def cairo_today(now: Optional[datetime] = None) -> date:
    return (now or datetime.now(_CAIRO)).astimezone(_CAIRO).date()


class DailyCounter:
    """take(key, limit) → True and counted, or False (nothing counted) once the key has used `limit` today."""

    def __init__(self) -> None:
        self._day: Optional[date] = None
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def take(self, key: str, limit: int, now: Optional[datetime] = None) -> bool:
        day = cairo_today(now)
        with self._lock:
            if day != self._day:
                self._day = day
                self._counts.clear()
            used = self._counts.get(key, 0)
            if used >= limit:
                return False
            self._counts[key] = used + 1
            return True

    def used(self, key: str) -> int:
        with self._lock:
            return self._counts.get(key, 0)


COPILOT_TURNS = DailyCounter()
PLANS = DailyCounter()


def user_key(headers: Mapping[str, str], client_host: Optional[str], device_id: Optional[str] = None) -> str:
    """The limit's key: the device id (body or header), else the caller's address."""
    dev = (device_id or headers.get(DEVICE_HEADER) or "").strip()
    if 8 <= len(dev) <= 64:
        return "dev:" + dev
    fwd = (headers.get("x-forwarded-for") or "").split(",")[0].strip()
    return "ip:" + (fwd or client_host or "unknown")
