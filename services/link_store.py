"""
services/link_store.py — persistence for RouteMind short links and live trips.

Two implementations behind one interface:
  • SupabaseLinkStore  the real one (tables `shared_links`, `live_trips`;
                       migrations/2026-09-14_shared_links.sql)
  • MemoryLinkStore    used by the test-suite and as the automatic fallback
                       when SUPABASE_URL / SUPABASE_SERVICE_KEY are unset, so a
                       stateless deployment still serves every page and the
                       API degrades to "links live until restart" instead of 500.

Ids are short and unguessable: 8 URL-safe base64 chars = 48 bits of entropy.
Edit/live tokens are stored as SHA-256 hashes; the plain token is returned
exactly once, to the creator.
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("routemind.links")

ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def new_id(length: int = 8) -> str:
    return "".join(secrets.choice(ID_ALPHABET) for _ in range(length))


def new_token() -> str:
    return secrets.token_urlsafe(24)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


def parse_iso(s: Any) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


class LinkStore:
    """Interface. Every method is synchronous (the Supabase wrapper is)."""

    # ── short links ──
    def create_link(self, kind: str, link: Dict[str, Any], label: Optional[str], creator: Optional[str],
                    ttl_days: int) -> Dict[str, Any]:
        raise NotImplementedError

    def get_link(self, link_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def revoke_link(self, link_id: str, edit_token: str) -> bool:
        raise NotImplementedError

    # ── live trips ──
    def create_live(self, destination: Dict[str, Any], label: Optional[str], ttl_minutes: int) -> Dict[str, Any]:
        raise NotImplementedError

    def get_live(self, live_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def update_live(self, live_id: str, token: str, patch: Dict[str, Any]) -> bool:
        raise NotImplementedError


class MemoryLinkStore(LinkStore):
    def __init__(self) -> None:
        self._links: Dict[str, Dict[str, Any]] = {}
        self._live: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_link(self, kind, link, label, creator, ttl_days):
        with self._lock:
            lid = new_id()
            while lid in self._links:
                lid = new_id()
            tok = new_token()
            now = now_utc()
            row = {
                "id": lid, "kind": kind, "link": link, "label": label, "creator": creator,
                "edit_token_hash": token_hash(tok), "created_at": iso(now),
                "expires_at": iso(now + timedelta(days=ttl_days)) if ttl_days > 0 else None,
                "revoked": False,
            }
            self._links[lid] = row
            return {**row, "edit_token": tok}

    def get_link(self, link_id):
        return self._links.get(link_id)

    def revoke_link(self, link_id, edit_token):
        with self._lock:
            row = self._links.get(link_id)
            if not row or row["edit_token_hash"] != token_hash(edit_token):
                return False
            row["revoked"] = True
            return True

    def create_live(self, destination, label, ttl_minutes):
        with self._lock:
            lid = new_id(10)
            while lid in self._live:
                lid = new_id(10)
            tok = new_token()
            now = now_utc()
            row = {
                "id": lid, "label": label, "destination": destination, "token_hash": token_hash(tok),
                "position": None, "eta_seconds": None, "remaining_m": None, "ended": False,
                "created_at": iso(now), "updated_at": iso(now),
                "expires_at": iso(now + timedelta(minutes=ttl_minutes)),
            }
            self._live[lid] = row
            return {**row, "token": tok}

    def get_live(self, live_id):
        return self._live.get(live_id)

    def update_live(self, live_id, token, patch):
        with self._lock:
            row = self._live.get(live_id)
            if not row or row["token_hash"] != token_hash(token):
                return False
            row.update(patch)
            row["updated_at"] = iso(now_utc())
            return True


class SupabaseLinkStore(LinkStore):
    """Thin mapping onto services.supabase_client.SupabaseClient (REST wrapper)."""

    def __init__(self, client) -> None:
        self._c = client

    def create_link(self, kind, link, label, creator, ttl_days):
        tok = new_token()
        now = now_utc()
        for _ in range(3):
            lid = new_id()
            row = {
                "id": lid, "kind": kind, "link": link, "label": label, "creator": creator,
                "edit_token_hash": token_hash(tok), "created_at": iso(now),
                "expires_at": iso(now + timedelta(days=ttl_days)) if ttl_days > 0 else None,
                "revoked": False,
            }
            res = self._c.table("shared_links").insert(row)
            if not res.error:
                return {**row, "edit_token": tok}
            if "duplicate" not in str(res.error).lower():
                raise RuntimeError(f"shared_links insert failed: {res.error}")
        raise RuntimeError("could not allocate a short id")

    def get_link(self, link_id):
        res = self._c.table("shared_links").select("*").eq("id", link_id).limit(1).execute()
        if res.error:
            raise RuntimeError(f"shared_links select failed: {res.error}")
        data = res.data or []
        return data[0] if data else None

    def revoke_link(self, link_id, edit_token):
        row = self.get_link(link_id)
        if not row or row.get("edit_token_hash") != token_hash(edit_token):
            return False
        res = self._c.table("shared_links").eq("id", link_id).update({"revoked": True}).execute_update()
        return not res.error

    def create_live(self, destination, label, ttl_minutes):
        tok = new_token()
        now = now_utc()
        for _ in range(3):
            lid = new_id(10)
            row = {
                "id": lid, "label": label, "destination": destination, "token_hash": token_hash(tok),
                "position": None, "eta_seconds": None, "remaining_m": None, "ended": False,
                "created_at": iso(now), "updated_at": iso(now),
                "expires_at": iso(now + timedelta(minutes=ttl_minutes)),
            }
            res = self._c.table("live_trips").insert(row)
            if not res.error:
                return {**row, "token": tok}
            if "duplicate" not in str(res.error).lower():
                raise RuntimeError(f"live_trips insert failed: {res.error}")
        raise RuntimeError("could not allocate a live id")

    def get_live(self, live_id):
        res = self._c.table("live_trips").select("*").eq("id", live_id).limit(1).execute()
        if res.error:
            raise RuntimeError(f"live_trips select failed: {res.error}")
        data = res.data or []
        return data[0] if data else None

    def update_live(self, live_id, token, patch):
        row = self.get_live(live_id)
        if not row or row.get("token_hash") != token_hash(token):
            return False
        res = self._c.table("live_trips").eq("id", live_id).update({**patch, "updated_at": iso(now_utc())}).execute_update()
        return not res.error


_store: Optional[LinkStore] = None


def get_link_store() -> LinkStore:
    """Supabase when configured, memory otherwise (logged once)."""
    global _store
    if _store is not None:
        return _store
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
        from services.supabase_client import get_supabase
        client = get_supabase()
        if client is not None:
            _store = SupabaseLinkStore(client)
            return _store
    logger.warning("links: Supabase not configured — using the in-memory store (links live until restart)")
    _store = MemoryLinkStore()
    return _store


def set_link_store(store: Optional[LinkStore]) -> None:
    """Tests only."""
    global _store
    _store = store
