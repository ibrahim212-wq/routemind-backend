"""
api/links.py — RouteMind links: short links, the landing pages recipients
see, live trips, the third-party resolver, and the App Links / Universal
Links well-known files.

Routes (docs/deep-links-architecture.md §6, docs/deep-links-setup.md):

  JSON (the app)
    POST /api/links                   mint  {link, label?, client?}      → {id, url, app_url, edit_token, expires_at}
    GET  /api/links/{id}              resolve our id                      → {id, kind, link, label, created_at, expires_at}
    POST /api/links/{id}/revoke       X-Edit-Token
    POST /api/links/resolve           {url}  third-party short link       → {final_url, hops, link|null}
    POST /api/live                    {destination, label?, ttl_minutes?} → {id, url, token, expires_at}
    POST /api/live/{id}/progress      X-Live-Token {lat,lng,heading?,speed?,eta_seconds?,remaining_m?}
    POST /api/live/{id}/end           X-Live-Token
    GET  /api/live/{id}               viewer state

  HTML (anyone)
    GET  /p/{id}  /r/{id}             landing page (place / route)
    GET  /go?lat=&lng=&name=…         stateless landing page (no row needed)
    GET  /open?u=<url>                landing page for a wrapped third-party link
    GET  /l/{id}                      live-trip viewer
    GET  /api/link-preview.png?…     OG map thumbnail (Mapbox Static, token stays server-side)

  Well-known
    GET  /.well-known/assetlinks.json
    GET  /.well-known/apple-app-site-association   (+ /apple-app-site-association)

Config (env): LINK_BASE_URL (public base, default = request base), MAPBOX_TOKEN,
ANDROID_PACKAGE, ANDROID_SHA256_FINGERPRINTS (comma-separated), APPLE_TEAM_ID,
IOS_BUNDLE_ID, APP_STORE_ID, PLAY_STORE_URL, APP_STORE_URL.
"""
from __future__ import annotations

import logging
import os
import re
import struct
import time
import zlib
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from services import link_grammar as lg
from services.link_resolver import ResolveError, resolve as resolve_external
from services.link_store import get_link_store, now_utc, parse_iso

logger = logging.getLogger("routemind.links")
router = APIRouter(tags=["links"])

_TEMPLATES = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))
_TEMPLATES.env.autoescape = True

LIVE_MAX_MINUTES = 8 * 60

# Read at call time (not import time) so a deploy can change them without a code
# change and the test-suite can set them per test.
def ANDROID_PACKAGE() -> str: return os.environ.get("ANDROID_PACKAGE", "com.routemind.app")
def IOS_BUNDLE_ID() -> str: return os.environ.get("IOS_BUNDLE_ID", "com.routemind.app")
def APPLE_TEAM_ID() -> str: return os.environ.get("APPLE_TEAM_ID", "S8QB4VV633")
def APP_STORE_ID() -> str: return os.environ.get("APP_STORE_ID", "")
def DEFAULT_TTL_DAYS() -> int: return int(os.environ.get("LINK_TTL_DAYS", "365"))

OPEN_SCHEMES = ("http://", "https://", "geo:", "waze:", "comgooglemaps:", "maps:", "om:", "ge0:", "yandexmaps:",
                "yandexnavi:", "dgis:", "here-location:", "here-route:", "osmandmaps:", "google.navigation:")


# ── config helpers ────────────────────────────────────────────────────────────

def base_url(request: Request) -> str:
    env = os.environ.get("LINK_BASE_URL", "").strip().rstrip("/")
    if env:
        return env
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}"


def play_store_url(link_id: Optional[str] = None) -> str:
    base = os.environ.get("PLAY_STORE_URL", f"https://play.google.com/store/apps/details?id={ANDROID_PACKAGE()}")
    if link_id:
        ref = urlencode({"utm_source": "routemind_link", "link": link_id})
        return f"{base}&referrer={quote(ref, safe='')}"
    return base


def app_store_url() -> str:
    env = os.environ.get("APP_STORE_URL", "").strip()
    if env:
        return env
    if APP_STORE_ID():
        return f"https://apps.apple.com/app/id{APP_STORE_ID()}"
    return ""


def fingerprints() -> List[str]:
    raw = os.environ.get("ANDROID_SHA256_FINGERPRINTS", "")
    return [f.strip().upper() for f in raw.split(",") if f.strip()]


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── rate limiting (in-memory token bucket per IP; Cloud Run runs one worker) ──

class _Bucket:
    def __init__(self, rate_per_min: int, burst: int):
        self.rate = rate_per_min / 60.0
        self.burst = burst
        self.state: Dict[str, List[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self.state.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens < 1:
            self.state[key] = [tokens, now]
            return False
        self.state[key] = [tokens - 1, now]
        if len(self.state) > 10_000:
            self.state.clear()
        return True


_mint_bucket = _Bucket(rate_per_min=30, burst=20)
_resolve_bucket = _Bucket(rate_per_min=60, burst=30)
_live_bucket = _Bucket(rate_per_min=240, burst=60)


def _limit(bucket: _Bucket, request: Request) -> None:
    if not bucket.allow(client_ip(request)):
        raise HTTPException(status_code=429, detail={"error": "rate_limited"})


def _err(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status)


# ── models ────────────────────────────────────────────────────────────────────

class MintBody(BaseModel):
    link: Dict[str, Any]
    label: Optional[str] = Field(default=None, max_length=200)
    client: Optional[Dict[str, Any]] = None
    ttl_days: Optional[int] = Field(default=None, ge=1, le=3650)


class ResolveBody(BaseModel):
    url: str = Field(min_length=8, max_length=4096)


class LiveStartBody(BaseModel):
    destination: Dict[str, Any]
    label: Optional[str] = Field(default=None, max_length=120)
    ttl_minutes: int = Field(default=240, ge=5, le=LIVE_MAX_MINUTES)
    lang: Optional[str] = None


class LiveProgressBody(BaseModel):
    lat: float
    lng: float
    heading: Optional[float] = None
    speed: Optional[float] = None
    eta_seconds: Optional[int] = Field(default=None, ge=0, le=86400 * 3)
    remaining_m: Optional[int] = Field(default=None, ge=0)


# ── short links ───────────────────────────────────────────────────────────────

def _row_state(row: Dict[str, Any]) -> Optional[str]:
    if row.get("revoked"):
        return "revoked"
    exp = parse_iso(row.get("expires_at"))
    if exp and exp < now_utc():
        return "expired"
    return None


@router.post("/api/links", status_code=201)
async def mint_link(body: MintBody, request: Request):
    _limit(_mint_bucket, request)
    link = lg.normalise_link(body.link)
    if link is None:
        return _err(422, "invalid_link")
    kind = "route" if link["kind"] == "route" else "place"
    label = (body.label or "").strip()[:120] or lg.link_title(link)
    creator = (body.client or {}).get("platform") if isinstance(body.client, dict) else None
    try:
        row = get_link_store().create_link(kind, link, label, creator, body.ttl_days or DEFAULT_TTL_DAYS())
    except Exception as e:  # noqa: BLE001
        logger.error("mint failed: %s", e)
        return _err(503, "store_unavailable")
    seg = "r" if kind == "route" else "p"
    b = base_url(request)
    return {
        "id": row["id"],
        "url": f"{b}/{seg}/{row['id']}",
        "app_url": f"routemind://v1/{seg}/{row['id']}",
        "edit_token": row["edit_token"],
        "expires_at": row.get("expires_at"),
    }


@router.get("/api/links/{link_id}")
async def get_link(link_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", link_id):
        return _err(404, "not_found")
    try:
        row = get_link_store().get_link(link_id)
    except Exception as e:  # noqa: BLE001
        logger.error("get_link failed: %s", e)
        return _err(503, "store_unavailable")
    if not row:
        return _err(404, "not_found")
    state = _row_state(row)
    if state:
        return _err(410, state)
    return {
        "id": row["id"],
        "kind": row.get("kind"),
        "link": row.get("link"),
        "label": row.get("label"),
        "created_at": row.get("created_at"),
        "expires_at": row.get("expires_at"),
    }


@router.post("/api/links/{link_id}/revoke")
async def revoke_link(link_id: str, x_edit_token: str = Header(default="")):
    if not x_edit_token:
        return _err(401, "missing_token")
    try:
        ok = get_link_store().revoke_link(link_id, x_edit_token)
    except Exception as e:  # noqa: BLE001
        logger.error("revoke failed: %s", e)
        return _err(503, "store_unavailable")
    if not ok:
        return _err(403, "forbidden")
    return {"ok": True}


@router.post("/api/links/resolve")
async def resolve_link(body: ResolveBody, request: Request):
    _limit(_resolve_bucket, request)
    try:
        return await resolve_external(body.url)
    except ResolveError as e:
        code = e.code
        if code in ("host_not_allowed", "scheme_not_allowed", "no_host", "credentials_in_url", "port_not_allowed", "private_address"):
            return _err(400, code)
        if code == "timeout":
            return _err(504, "timeout")
        return _err(422, "unresolvable")


# ── live trips ────────────────────────────────────────────────────────────────

@router.post("/api/live", status_code=201)
async def start_live(body: LiveStartBody, request: Request):
    _limit(_mint_bucket, request)
    dest = lg.normalise_link(body.destination)
    if dest is None or lg.link_point(dest) is None:
        return _err(422, "invalid_destination")
    label = (body.label or "").strip()[:80] or None
    try:
        row = get_link_store().create_live(dest, label, body.ttl_minutes)
    except Exception as e:  # noqa: BLE001
        logger.error("live create failed: %s", e)
        return _err(503, "store_unavailable")
    return {"id": row["id"], "url": f"{base_url(request)}/l/{row['id']}", "token": row["token"], "expires_at": row["expires_at"]}


@router.post("/api/live/{live_id}/progress")
async def live_progress(live_id: str, body: LiveProgressBody, request: Request, x_live_token: str = Header(default="")):
    _limit(_live_bucket, request)
    pt = lg.valid_point(body.lat, body.lng)
    if not pt:
        return _err(422, "invalid_point")
    patch = {
        "position": {"lat": pt[0], "lng": pt[1], "heading": body.heading, "updated_at": lg_iso_now()},
        "eta_seconds": body.eta_seconds,
        "remaining_m": body.remaining_m,
    }
    try:
        ok = get_link_store().update_live(live_id, x_live_token, patch)
    except Exception as e:  # noqa: BLE001
        logger.error("live progress failed: %s", e)
        return _err(503, "store_unavailable")
    if not ok:
        return _err(403, "forbidden")
    return {"ok": True}


@router.post("/api/live/{live_id}/end")
async def live_end(live_id: str, request: Request, x_live_token: str = Header(default="")):
    try:
        ok = get_link_store().update_live(live_id, x_live_token, {"ended": True})
    except Exception as e:  # noqa: BLE001
        logger.error("live end failed: %s", e)
        return _err(503, "store_unavailable")
    if not ok:
        return _err(403, "forbidden")
    return {"ok": True}


def lg_iso_now() -> str:
    from services.link_store import iso
    return iso(now_utc()) or ""


def _live_view(row: Dict[str, Any]) -> Dict[str, Any]:
    exp = parse_iso(row.get("expires_at"))
    expired = bool(exp and exp < now_utc())
    return {
        "id": row["id"],
        "label": row.get("label"),
        "destination": row.get("destination"),
        "position": row.get("position"),
        "eta_seconds": row.get("eta_seconds"),
        "remaining_m": row.get("remaining_m"),
        "ended": bool(row.get("ended")) or expired,
        "expires_at": row.get("expires_at"),
    }


@router.get("/api/live/{live_id}")
async def get_live(live_id: str):
    try:
        row = get_link_store().get_live(live_id)
    except Exception as e:  # noqa: BLE001
        logger.error("live get failed: %s", e)
        return _err(503, "store_unavailable")
    if not row:
        return _err(404, "not_found")
    exp = parse_iso(row.get("expires_at"))
    if exp and exp < now_utc() and not row.get("ended"):
        return _err(410, "expired")
    return _live_view(row)


# ── landing pages ─────────────────────────────────────────────────────────────

def _lang(request: Request) -> str:
    q = request.query_params.get("lang", "").lower()
    if q in ("ar", "en"):
        return q
    al = request.headers.get("accept-language", "").lower()
    return "ar" if al.startswith("ar") or ",ar" in al or " ar" in al else "en"


def _security_headers(resp: Response, nonce: str) -> Response:
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: https:; style-src 'unsafe-inline'; "
        f"script-src 'nonce-{nonce}'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    )
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp


def _render(request: Request, link: Optional[Dict[str, Any]], *, link_id: Optional[str], seg: str,
            state: Optional[str], status: int = 200, wrapped_url: Optional[str] = None) -> Response:
    import secrets
    lang = _lang(request)
    nonce = secrets.token_urlsafe(12)
    b = base_url(request)
    pt = lg.link_point(link) if link else None
    title = lg.link_title(link, lang) if link else ("RouteMind" if not state else _state_title(state, lang))
    coords = f"{lg.fmt(pt[0])}, {lg.fmt(pt[1])}" if pt else ""
    page_url = f"{b}/{seg}/{link_id}" if link_id else str(request.url)
    if link_id:
        app_url = f"routemind://v1/{'p' if seg == 'p' else ('r' if seg == 'r' else seg)}/{link_id}"
    elif wrapped_url:
        app_url = "routemind://v1/open?" + urlencode({"u": wrapped_url})
    elif link:
        app_url = "routemind://v1/" + ("route" if link["kind"] == "route" else ("q" if link["kind"] == "query" else "place")) + "?" + urlencode(_go_params(link))
    else:
        app_url = "routemind://v1/place"
    intent_url = None
    if app_url.startswith("routemind://"):
        rest = app_url[len("routemind://"):]
        intent_url = (f"intent://{rest}#Intent;scheme=routemind;package={ANDROID_PACKAGE()};"
                      f"S.browser_fallback_url={quote(play_store_url(link_id), safe='')};end")
    address = link.get("address") if link and link.get("kind") == "place" else None
    via = len(link.get("waypoints") or []) if link and link.get("kind") == "route" else 0
    ext = _external_urls(link) if link else {}
    ctx = {
        "request": request,
        "lang": lang,
        "rtl": lang == "ar",
        "nonce": nonce,
        "title": title,
        "address": address,
        "coords": coords,
        "lat": lg.fmt(pt[0]) if pt else "",
        "lng": lg.fmt(pt[1]) if pt else "",
        "via": via,
        "state": state,
        "state_text": _state_text(state, lang) if state else "",
        "page_url": page_url,
        "app_url": app_url,
        "intent_url": intent_url,
        "play_url": play_store_url(link_id),
        "appstore_url": app_store_url(),
        "appstore_id": APP_STORE_ID(),
        "og_image": f"{b}/api/link-preview.png?" + urlencode({"lat": lg.fmt(pt[0]), "lng": lg.fmt(pt[1])}) if pt else f"{b}/api/link-preview.png",
        "ext": ext,
        "t": _T[lang],
        "kind": (link or {}).get("kind"),
        "wrapped_url": wrapped_url,
    }
    resp = _TEMPLATES.TemplateResponse("landing.html", ctx, status_code=status)
    return _security_headers(resp, nonce)


def _go_params(link: Dict[str, Any]) -> Dict[str, str]:
    pt = lg.link_point(link)
    out: Dict[str, str] = {}
    if link["kind"] in ("coordinate", "place") and pt:
        out = {"lat": lg.fmt(pt[0]), "lng": lg.fmt(pt[1])}
        if link.get("label"):
            out["name"] = link["label"]
        if link.get("address"):
            out["addr"] = link["address"]
    elif link["kind"] == "query":
        out = {"text": link.get("query", "")}
    elif link["kind"] == "route":
        def stop(s):
            p = s.get("point")
            return f"{lg.fmt(p['lat'])},{lg.fmt(p['lng'])}" if p else (s.get("query") or "")
        out = {"d": stop(link["destination"])}
        if link["destination"].get("label"):
            out["dn"] = link["destination"]["label"]
        if link.get("origin"):
            out["o"] = stop(link["origin"])
        if link.get("waypoints"):
            out["w"] = "|".join(stop(w) for w in link["waypoints"])
        if link.get("profile"):
            out["mode"] = {"driving": "drive", "walking": "walk", "cycling": "bike", "transit": "transit"}[link["profile"]]
    if link.get("navigate"):
        out["nav"] = "1"
    return out


def _external_urls(link: Dict[str, Any]) -> Dict[str, str]:
    """Google / Apple / Waze twins (same shapes the app's ShareComposer emits)."""
    pt = lg.link_point(link)
    if not pt:
        q = link.get("query") or link.get("label") or ""
        return {
            "google": "https://www.google.com/maps/search/?" + urlencode({"api": "1", "query": q}),
            "apple": "https://maps.apple.com/?" + urlencode({"q": q}),
            "waze": "https://waze.com/ul?" + urlencode({"q": q, "navigate": "yes"}),
        }
    ll = f"{lg.fmt(pt[0])},{lg.fmt(pt[1])}"
    if link.get("kind") == "route":
        g: Dict[str, str] = {"api": "1", "destination": ll, "travelmode": {"walking": "walking", "cycling": "bicycling", "transit": "transit"}.get(link.get("profile") or "", "driving")}
        o = (link.get("origin") or {}).get("point")
        if o:
            g["origin"] = f"{lg.fmt(o['lat'])},{lg.fmt(o['lng'])}"
        wps = [w["point"] for w in link.get("waypoints") or [] if w.get("point")]
        if wps:
            g["waypoints"] = "|".join(f"{lg.fmt(w['lat'])},{lg.fmt(w['lng'])}" for w in wps)
        a = {"daddr": ll, "dirflg": "w" if link.get("profile") == "walking" else "d"}
        if o:
            a["saddr"] = g["origin"]
        return {
            "google": "https://www.google.com/maps/dir/?" + urlencode(g),
            "apple": "https://maps.apple.com/?" + urlencode(a),
            "waze": "https://waze.com/ul?" + urlencode({"ll": ll, "navigate": "yes"}),
        }
    name = link.get("label") or ""
    return {
        "google": "https://www.google.com/maps/search/?" + urlencode({"api": "1", "query": ll}),
        "apple": "https://maps.apple.com/?" + urlencode({"ll": ll, **({"q": name} if name else {})}),
        "waze": "https://waze.com/ul?" + urlencode({"ll": ll, "navigate": "yes"}),
    }


_T = {
    "en": {
        "open": "Open in RouteMind", "get": "Get RouteMind", "google": "Google Maps", "apple": "Apple Maps",
        "waze": "Waze", "copy": "Copy coordinates", "copied": "Copied", "shared": "Shared location",
        "via": "via {n} stops", "route": "Route", "tag": "Smart traffic navigation for Cairo & Giza",
        "no_app": "Don't have the app yet?", "opening": "Opening RouteMind…", "still": "Still here? Install RouteMind:",
        "live": "Live trip", "arriving": "Arriving in {m} min", "updated": "Updated {s}s ago", "ended": "This trip has ended.",
        "waiting": "Waiting for the first position…", "heading": "Heading to {d}", "unavailable": "This link is no longer available.",
    },
    "ar": {
        "open": "افتح في RouteMind", "get": "نزّل RouteMind", "google": "جوجل مابس", "apple": "آبل مابس",
        "waze": "ويز", "copy": "نسخ الإحداثيات", "copied": "اتنسخ", "shared": "مكان اتبعتلك",
        "via": "عن طريق {n} محطات", "route": "مسار", "tag": "ملاحة ذكية بالمرور للقاهرة والجيزة",
        "no_app": "لسه منزّلتش التطبيق؟", "opening": "بنفتح RouteMind…", "still": "لسه هنا؟ نزّل RouteMind:",
        "live": "رحلة لايف", "arriving": "هيوصل بعد {m} دقيقة", "updated": "اتحدّث من {s} ثانية", "ended": "الرحلة دي خلصت.",
        "waiting": "مستنيين أول موقع…", "heading": "رايح {d}", "unavailable": "اللينك ده مبقاش شغال.",
    },
}


def _state_title(state: str, lang: str) -> str:
    return {"en": {"expired": "This link has expired", "revoked": "The sender removed this link", "not_found": "Link not found"},
            "ar": {"expired": "اللينك ده انتهى", "revoked": "اللي بعت اللينك لغاه", "not_found": "اللينك مش موجود"}}[lang].get(state, "RouteMind")


def _state_text(state: str, lang: str) -> str:
    return f"{_state_title(state, lang)} · {_T[lang]['unavailable']}"


def _load_row(link_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", link_id):
        return None, "not_found"
    try:
        row = get_link_store().get_link(link_id)
    except Exception as e:  # noqa: BLE001
        logger.error("landing get failed: %s", e)
        return None, "unavailable"
    if not row:
        return None, "not_found"
    return row, _row_state(row)


@router.get("/p/{link_id}", response_class=HTMLResponse)
async def landing_place(link_id: str, request: Request):
    row, state = _load_row(link_id)
    link = row.get("link") if row else None
    status = 200 if not state else (410 if state in ("expired", "revoked") else 404)
    # An expired row still renders its coordinates (the pin is not a secret); the state banner says why.
    return _render(request, link, link_id=link_id, seg="p", state=state, status=status)


@router.get("/r/{link_id}", response_class=HTMLResponse)
async def landing_route(link_id: str, request: Request):
    row, state = _load_row(link_id)
    link = row.get("link") if row else None
    status = 200 if not state else (410 if state in ("expired", "revoked") else 404)
    return _render(request, link, link_id=link_id, seg="r", state=state, status=status)


@router.get("/go", response_class=HTMLResponse)
async def landing_go(request: Request):
    q = {k.lower(): v for k, v in request.query_params.items()}
    link = lg.link_from_go_params(q)
    if link is None:
        return _render(request, None, link_id=None, seg="go", state="not_found", status=404)
    return _render(request, link, link_id=None, seg="go", state=None)


@router.get("/open", response_class=HTMLResponse)
async def landing_open(request: Request):
    u = request.query_params.get("u", "").strip()
    if not u.lower().startswith(OPEN_SCHEMES) or len(u) > 2048:
        return _render(request, None, link_id=None, seg="open", state="not_found", status=404)
    return _render(request, None, link_id=None, seg="open", state=None, wrapped_url=u)


@router.get("/l/{live_id}", response_class=HTMLResponse)
async def landing_live(live_id: str, request: Request):
    import secrets
    lang = _lang(request)
    nonce = secrets.token_urlsafe(12)
    try:
        row = get_link_store().get_live(live_id) if re.fullmatch(r"[A-Za-z0-9_-]{4,64}", live_id) else None
    except Exception:  # noqa: BLE001
        row = None
    view = _live_view(row) if row else None
    dest = view.get("destination") if view else None
    dpt = lg.link_point(dest) if dest else None
    ctx = {
        "request": request, "lang": lang, "rtl": lang == "ar", "nonce": nonce, "t": _T[lang],
        "live_id": live_id, "title": (view or {}).get("label") or _T[lang]["live"],
        "dest_title": lg.link_title(dest, lang) if dest else "",
        "dest_lat": lg.fmt(dpt[0]) if dpt else "", "dest_lng": lg.fmt(dpt[1]) if dpt else "",
        "found": view is not None, "ended": bool(view and view["ended"]),
        "page_url": f"{base_url(request)}/l/{live_id}",
        "app_url": f"routemind://v1/live/{live_id}",
        "play_url": play_store_url(None), "appstore_url": app_store_url(), "appstore_id": APP_STORE_ID(),
        "og_image": f"{base_url(request)}/api/link-preview.png?" + (urlencode({"lat": lg.fmt(dpt[0]), "lng": lg.fmt(dpt[1])}) if dpt else ""),
    }
    resp = _TEMPLATES.TemplateResponse("live.html", ctx, status_code=200 if view else 404)
    return _security_headers(resp, nonce)


# ── OG image (Mapbox Static Images proxied; token never leaves the server) ────

_PREVIEW_CACHE: Dict[str, tuple] = {}


def _fallback_png(w: int = 1200, h: int = 630) -> bytes:
    """A solid brand-navy PNG built with zlib only — used when Mapbox is unavailable."""
    row = b"\x00" + bytes([0x06, 0x13, 0x24] * w)
    raw = row * h
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


@router.get("/api/link-preview.png")
async def preview_png(request: Request):
    pt = lg.valid_point(request.query_params.get("lat"), request.query_params.get("lng"))
    token = os.environ.get("MAPBOX_TOKEN", "")
    headers = {"Cache-Control": "public, max-age=86400"}
    if not pt or not token:
        return Response(content=_fallback_png(), media_type="image/png", headers=headers)
    key = f"{lg.fmt(pt[0])},{lg.fmt(pt[1])}"
    cached = _PREVIEW_CACHE.get(key)
    if cached and cached[0] > time.monotonic():
        return Response(content=cached[1], media_type="image/png", headers=headers)
    lng, lat = lg.fmt(pt[1]), lg.fmt(pt[0])
    url = (f"https://api.mapbox.com/styles/v1/mapbox/streets-v12/static/"
           f"pin-l+3aa76d({lng},{lat})/{lng},{lat},15,0/1200x630@2x?access_token={token}&attribution=false&logo=false")
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
            if len(_PREVIEW_CACHE) > 500:
                _PREVIEW_CACHE.clear()
            _PREVIEW_CACHE[key] = (time.monotonic() + 3600, r.content)
            return Response(content=r.content, media_type="image/png", headers=headers)
        logger.warning("mapbox static %s: %s", r.status_code, r.text[:120])
    except httpx.HTTPError as e:
        logger.warning("mapbox static failed: %s", e)
    return Response(content=_fallback_png(), media_type="image/png", headers=headers)


# ── well-known ────────────────────────────────────────────────────────────────

@router.get("/.well-known/assetlinks.json")
async def assetlinks():
    fps = fingerprints()
    body = [{
        "relation": ["delegate_permission/common.handle_all_urls"],
        "target": {"namespace": "android_app", "package_name": ANDROID_PACKAGE(), "sha256_cert_fingerprints": fps},
    }]
    # Content-Type must be application/json; no redirect; no auth (Android's verifier is strict).
    return JSONResponse(body, media_type="application/json", headers={"Cache-Control": "public, max-age=3600"})


def _aasa() -> Dict[str, Any]:
    app_id = f"{APPLE_TEAM_ID()}.{IOS_BUNDLE_ID()}"
    return {
        "applinks": {
            "details": [{
                "appIDs": [app_id],
                "components": [
                    {"/": "/p/*", "comment": "place short link"},
                    {"/": "/r/*", "comment": "route short link"},
                    {"/": "/l/*", "comment": "live trip"},
                    {"/": "/go", "comment": "stateless link"},
                    {"/": "/go/*"},
                    {"/": "/open", "comment": "wrapped third-party link"},
                    {"/": "/.well-known/*", "exclude": True},
                    {"/": "/api/*", "exclude": True},
                ],
            }],
        },
        "webcredentials": {"apps": [app_id]},
    }


@router.get("/.well-known/apple-app-site-association")
async def aasa_wellknown():
    return JSONResponse(_aasa(), media_type="application/json", headers={"Cache-Control": "public, max-age=3600"})


@router.get("/apple-app-site-association")
async def aasa_root():
    return JSONResponse(_aasa(), media_type="application/json", headers={"Cache-Control": "public, max-age=3600"})
