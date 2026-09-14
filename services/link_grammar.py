"""
services/link_grammar.py — the server-side half of RouteMind's link grammar.

The app's Dart parser (lib/features/deeplink/domain/) is THE interpreter of
inbound links. This module only needs to:

  • validate a link payload the app posts when minting a short link
    (kinds: coordinate | place | query | route) and normalise it;
  • read the stateless `/go?lat=&lng=&name=…` query (same parameter names
    the Dart RouteMindLink builder writes) so the landing page can render it
    with no database row;
  • extract coordinates from a fetched HTML page for the third-party
    resolver (Google's !3d!4d blob, og:image center=, @lat,lng viewport,
    JSON "lat"/"lng", place:location meta) — the same regexes the Dart
    client falls back to, so both sides agree.

Pure Python, no I/O.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, Optional, Tuple

LAT_MIN, LAT_MAX, LNG_MIN, LNG_MAX = -90.0, 90.0, -180.0, 180.0
MAX_LABEL = 120
MAX_QUERY = 200


def valid_point(lat: Any, lng: Any) -> Optional[Tuple[float, float]]:
    try:
        la, lo = float(lat), float(lng)
    except (TypeError, ValueError):
        return None
    if la != la or lo != lo:  # NaN
        return None
    if not (LAT_MIN <= la <= LAT_MAX and LNG_MIN <= lo <= LNG_MAX):
        return None
    return la, lo


def _clean_text(v: Any, limit: int) -> Optional[str]:
    if v is None:
        return None
    s = str(v).replace("\r", " ").replace("\n", " ").strip()
    # strip bidi/zero-width controls that break the card layout
    s = re.sub(r"[​-‏‪-‮⁠-⁤⁦-⁩﻿]", "", s)
    if not s:
        return None
    return s[:limit]


def _stop(d: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(d, dict):
        return None
    out: Dict[str, Any] = {}
    p = d.get("point")
    if isinstance(p, dict):
        pt = valid_point(p.get("lat"), p.get("lng"))
        if pt:
            out["point"] = {"lat": pt[0], "lng": pt[1]}
    q = _clean_text(d.get("query"), MAX_QUERY)
    if q:
        out["query"] = q
    lab = _clean_text(d.get("label"), MAX_LABEL)
    if lab:
        out["label"] = lab
    pid = _clean_text(d.get("placeId"), 200)
    if pid:
        out["placeId"] = pid
    if "point" not in out and "query" not in out and "placeId" not in out:
        return None
    return out


def normalise_link(link: Any) -> Optional[Dict[str, Any]]:
    """The subset of the Dart LocationLink JSON we store and render. None = reject."""
    if not isinstance(link, dict):
        return None
    kind = link.get("kind")
    nav = bool(link.get("navigate"))
    hint = _clean_text(link.get("hintLabel"), MAX_LABEL)
    if kind in ("coordinate", "place"):
        p = link.get("point") if isinstance(link.get("point"), dict) else {}
        pt = valid_point(p.get("lat"), p.get("lng"))
        if not pt:
            return None
        out: Dict[str, Any] = {"kind": kind, "point": {"lat": pt[0], "lng": pt[1]}}
        if kind == "place":
            out["label"] = _clean_text(link.get("label"), MAX_LABEL) or ""
            addr = _clean_text(link.get("address"), MAX_QUERY)
            if addr:
                out["address"] = addr
            pid = _clean_text(link.get("placeId"), 200)
            if pid:
                out["placeId"] = pid
        else:
            z = link.get("zoom")
            try:
                if z is not None and 0 < float(z) <= 24:
                    out["zoom"] = float(z)
            except (TypeError, ValueError):
                pass
    elif kind == "query":
        q = _clean_text(link.get("query"), MAX_QUERY)
        pid = _clean_text(link.get("placeId"), 200)
        if not q and not pid:
            return None
        out = {"kind": "query", "query": q or ""}
        if pid:
            out["placeId"] = pid
        near = link.get("near")
        if isinstance(near, dict):
            pt = valid_point(near.get("lat"), near.get("lng"))
            if pt:
                out["near"] = {"lat": pt[0], "lng": pt[1]}
    elif kind == "route":
        dest = _stop(link.get("destination"))
        if not dest:
            return None
        out = {"kind": "route", "destination": dest}
        origin = _stop(link.get("origin"))
        if origin:
            out["origin"] = origin
        wps = [w for w in (_stop(x) for x in (link.get("waypoints") or [])[:25]) if w]
        if wps:
            out["waypoints"] = wps
        prof = link.get("profile")
        if prof in ("driving", "walking", "cycling", "transit"):
            out["profile"] = prof
    else:
        return None
    out["source"] = "routemind"
    if nav:
        out["navigate"] = True
    if hint:
        out["hintLabel"] = hint
    return out


def link_point(link: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """The point a link is about (destination for routes)."""
    if link.get("kind") in ("coordinate", "place"):
        p = link.get("point") or {}
        return valid_point(p.get("lat"), p.get("lng"))
    if link.get("kind") == "route":
        p = (link.get("destination") or {}).get("point") or {}
        return valid_point(p.get("lat"), p.get("lng"))
    return None


def link_title(link: Dict[str, Any], lang: str = "en") -> str:
    if link.get("kind") == "place" and link.get("label"):
        return link["label"]
    if link.get("kind") == "query" and link.get("query"):
        return link["query"]
    if link.get("kind") == "route":
        d = link.get("destination") or {}
        return d.get("label") or d.get("query") or ("Route" if lang != "ar" else "مسار")
    h = link.get("hintLabel")
    if h:
        return h
    return "Shared location" if lang != "ar" else "مكان اتبعتلك"


# ── /go?… (stateless) ─────────────────────────────────────────────────────────

def _pair(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        return None
    return valid_point(parts[0].strip(), parts[1].strip())


def link_from_go_params(q: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Mirror of RouteMindLink.parse for the https /go form. Keys are lower-case."""
    kind = q.get("kind")
    nav = q.get("nav") in ("1", "true", "yes")
    if kind == "route" or "d" in q:
        d = q.get("d") or q.get("dest") or q.get("destination")
        if not d:
            return None
        def stop(v: str, name: Optional[str]) -> Dict[str, Any]:
            pt = _pair(v)
            s: Dict[str, Any] = {"point": {"lat": pt[0], "lng": pt[1]}} if pt else {"query": _clean_text(v, MAX_QUERY) or ""}
            n = _clean_text(name, MAX_LABEL)
            if n:
                s["label"] = n
            return s
        out: Dict[str, Any] = {"kind": "route", "destination": stop(d, q.get("dn"))}
        o = q.get("o") or q.get("origin")
        if o:
            out["origin"] = stop(o, q.get("on"))
        w = [x for x in (q.get("w") or "").split("|") if x.strip()]
        wn = (q.get("wn") or "").split("|")
        if w:
            out["waypoints"] = [stop(x, wn[i] if i < len(wn) else None) for i, x in enumerate(w[:25])]
        mode = {"drive": "driving", "walk": "walking", "bike": "cycling", "transit": "transit"}.get(q.get("mode") or "")
        if mode:
            out["profile"] = mode
    elif kind == "q" or "text" in q:
        t = _clean_text(q.get("text") or q.get("q"), MAX_QUERY)
        if not t and not q.get("pid"):
            return None
        out = {"kind": "query", "query": t or ""}
        if q.get("pid"):
            out["placeId"] = q["pid"]
        near = _pair(q.get("near"))
        if near:
            out["near"] = {"lat": near[0], "lng": near[1]}
    else:
        pt = valid_point(q.get("lat"), q.get("lng")) or _pair(q.get("ll"))
        name = _clean_text(q.get("name") or q.get("n"), MAX_LABEL)
        if not pt:
            if name:
                out = {"kind": "query", "query": name}
            else:
                return None
        elif name or q.get("addr") or q.get("pid"):
            out = {"kind": "place", "point": {"lat": pt[0], "lng": pt[1]}, "label": name or ""}
            addr = _clean_text(q.get("addr") or q.get("address"), MAX_QUERY)
            if addr:
                out["address"] = addr
            if q.get("pid"):
                out["placeId"] = q["pid"]
        else:
            out = {"kind": "coordinate", "point": {"lat": pt[0], "lng": pt[1]}}
            try:
                z = float(q.get("z") or "")
                if 0 < z <= 24:
                    out["zoom"] = z
            except ValueError:
                pass
    out["source"] = "routemind"
    if nav:
        out["navigate"] = True
    return out


# ── HTML extraction for the resolver ──────────────────────────────────────────

_NUM = r"(-?\d{1,3}\.\d+)"
_RE_3D4D = re.compile(r"!3d" + _NUM + r"!4d" + _NUM)
_RE_CENTER = re.compile(r"center=" + _NUM + r"(?:%2C|,)" + _NUM, re.I)
_RE_AT = re.compile(r"@" + _NUM + r"," + _NUM + r",\d{1,2}(?:\.\d+)?z")
_RE_JSON = re.compile(r'"lat(?:itude)?"\s*:\s*' + _NUM + r'\s*,\s*"l(?:ng|on|ongitude)"\s*:\s*' + _NUM)
_RE_META_LAT = re.compile(r'<meta[^>]+(?:property|name)="(?:place:location:latitude|og:latitude)"[^>]+content="' + _NUM + '"', re.I)
_RE_META_LNG = re.compile(r'<meta[^>]+(?:property|name)="(?:place:location:longitude|og:longitude)"[^>]+content="' + _NUM + '"', re.I)
_RE_OG_TITLE = re.compile(r'<meta[^>]+property="og:title"[^>]+content="([^"]*)"', re.I)
_RE_TITLE = re.compile(r"<title[^>]*>([^<]{1,200})</title>", re.I)


def extract_from_html(page: str) -> Optional[Dict[str, Any]]:
    """A coordinate/place link found in a page, or None."""
    if not page:
        return None
    pt = None
    m = _RE_3D4D.search(page)
    if m:
        pt = valid_point(m.group(1), m.group(2))
    if not pt:
        la, lo = _RE_META_LAT.search(page), _RE_META_LNG.search(page)
        if la and lo:
            pt = valid_point(la.group(1), lo.group(1))
    if not pt:
        m = _RE_CENTER.search(page)
        if m:
            pt = valid_point(m.group(1), m.group(2))
    if not pt:
        m = _RE_AT.search(page)
        if m:
            pt = valid_point(m.group(1), m.group(2))
    if not pt:
        m = _RE_JSON.search(page)
        if m:
            pt = valid_point(m.group(1), m.group(2))
    if not pt:
        return None
    title = None
    m = _RE_OG_TITLE.search(page) or _RE_TITLE.search(page)
    if m:
        t = html.unescape(m.group(1)).strip()
        t = re.sub(r"\s*[-–|·]\s*Google Maps.*$", "", t).strip()
        title = _clean_text(t, MAX_LABEL)
    if title:
        return {"kind": "place", "point": {"lat": pt[0], "lng": pt[1]}, "label": title, "source": "shortener"}
    return {"kind": "coordinate", "point": {"lat": pt[0], "lng": pt[1]}, "source": "shortener"}


def fmt(v: float) -> str:
    """6 decimals, trailing zeros trimmed — the canonical human form."""
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"
