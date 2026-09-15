"""
tests/test_links.py — the link service: mint/resolve/revoke, landing pages
(XSS-safe, RTL, OG), stateless /go, live trips, the third-party resolver's
SSRF guards and page extraction, and the well-known files.

Runs against the in-memory store; the resolver's network is mocked at the
httpx transport layer so nothing here touches the internet.
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.links import router
from services import link_grammar as lg
from services import link_resolver as lr
from services.link_store import MemoryLinkStore, set_link_store


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ANDROID_SHA256_FINGERPRINTS", "AA:BB:CC, dd:ee:ff")
    monkeypatch.setenv("APPLE_TEAM_ID", "R3D75N7XCX")
    monkeypatch.setenv("APP_STORE_ID", "123456789")
    monkeypatch.delenv("LINK_BASE_URL", raising=False)
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    set_link_store(MemoryLinkStore())
    app = FastAPI()
    app.include_router(router)
    with TestClient(app, base_url="https://links.test") as c:
        yield c
    set_link_store(None)


PLACE = {"kind": "place", "source": "routemind", "point": {"lat": 30.0459, "lng": 31.2243},
         "label": "برج القاهرة", "address": "Zamalek, Cairo"}
ROUTE = {"kind": "route", "source": "routemind",
         "destination": {"point": {"lat": 29.9792, "lng": 31.1342}, "label": "Pyramids"},
         "origin": {"point": {"lat": 30.0444, "lng": 31.2357}},
         "waypoints": [{"point": {"lat": 30.01, "lng": 31.2}}], "profile": "driving", "navigate": True}


# ── grammar ───────────────────────────────────────────────────────────────────

def test_normalise_rejects_bad_points_and_kinds():
    assert lg.normalise_link({"kind": "place", "point": {"lat": 95, "lng": 31}}) is None
    assert lg.normalise_link({"kind": "coordinate", "point": {"lat": "x", "lng": 31}}) is None
    assert lg.normalise_link({"kind": "live", "id": "x"}) is None
    assert lg.normalise_link({"kind": "route", "destination": {}}) is None
    assert lg.normalise_link("nope") is None


def test_normalise_keeps_precision_and_strips_controls():
    n = lg.normalise_link({"kind": "place", "point": {"lat": 30.044412345678, "lng": 31.235767891234},
                           "label": "‏كافيه‎", "address": "  x\ny  "})
    assert n["point"] == {"lat": 30.044412345678, "lng": 31.235767891234}
    assert n["label"] == "كافيه"
    assert n["address"] == "x y"


def test_go_params_mirror_the_app_grammar():
    assert lg.link_from_go_params({"lat": "30.0444", "lng": "31.2357", "name": "Cafe"})["kind"] == "place"
    c = lg.link_from_go_params({"lat": "30.0444", "lng": "31.2357", "z": "16"})
    assert c["kind"] == "coordinate" and c["zoom"] == 16.0
    r = lg.link_from_go_params({"kind": "route", "d": "29.9792,31.1342", "dn": "Pyramids", "o": "30.0444,31.2357",
                                "w": "30.01,31.2|30.02,31.21", "mode": "walk", "nav": "1"})
    assert r["kind"] == "route" and r["profile"] == "walking" and len(r["waypoints"]) == 2 and r["navigate"] is True
    assert lg.link_from_go_params({"text": "Cairo Tower"})["kind"] == "query"
    assert lg.link_from_go_params({"lat": "999", "lng": "1"}) is None
    assert lg.link_from_go_params({}) is None


def test_html_extraction_priorities():
    page = '<meta property="og:title" content="Cafe &amp; Co - Google Maps"><meta property="og:image" content="https://maps.google.com/maps/api/staticmap?center=30.1%2C31.2&zoom=15">' \
           'data=!3d30.0459!4d31.2243'
    l = lg.extract_from_html(page)
    assert l["point"] == {"lat": 30.0459, "lng": 31.2243} and l["label"] == "Cafe & Co"
    only_center = lg.extract_from_html('<meta property="og:image" content="x?center=30.1%2C31.2&z=1">')
    assert only_center["point"] == {"lat": 30.1, "lng": 31.2} and only_center["kind"] == "coordinate"
    assert lg.extract_from_html("<html>nothing here</html>") is None


# ── mint / get / revoke ───────────────────────────────────────────────────────

def test_mint_get_revoke_cycle(client):
    r = client.post("/api/links", json={"link": PLACE, "label": "Cairo Tower", "client": {"platform": "android"}})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["url"].startswith("https://links.test/p/") and body["app_url"].startswith("routemind://v1/p/")
    lid = body["id"]
    assert len(lid) == 8 and body["edit_token"]

    g = client.get(f"/api/links/{lid}")
    assert g.status_code == 200
    assert g.json()["link"]["point"] == PLACE["point"]
    assert "edit_token" not in g.json() and "edit_token_hash" not in g.json()

    assert client.post(f"/api/links/{lid}/revoke", headers={"X-Edit-Token": "wrong"}).status_code == 403
    assert client.post(f"/api/links/{lid}/revoke", headers={"X-Edit-Token": body["edit_token"]}).status_code == 200
    g2 = client.get(f"/api/links/{lid}")
    assert g2.status_code == 410 and g2.json()["error"] == "revoked"


def test_mint_route_lands_on_r(client):
    r = client.post("/api/links", json={"link": ROUTE})
    assert r.status_code == 201
    assert "/r/" in r.json()["url"]


def test_mint_rejects_garbage(client):
    assert client.post("/api/links", json={"link": {"kind": "place", "point": {"lat": 200, "lng": 0}}}).status_code == 422
    assert client.post("/api/links", json={"link": "x"}).status_code == 422


def test_unknown_and_bad_ids(client):
    assert client.get("/api/links/doesnotexist").status_code == 404
    assert client.get("/api/links/!!").status_code == 404


# ── landing pages ─────────────────────────────────────────────────────────────

def test_landing_page_is_xss_safe_rtl_and_rich(client):
    evil = {"kind": "place", "point": {"lat": 30.0444, "lng": 31.2357},
            "label": '<script>alert(1)</script>"><img src=x onerror=alert(2)>', "address": "الدقي، الجيزة"}
    lid = client.post("/api/links", json={"link": evil}).json()["id"]
    page = client.get(f"/p/{lid}", headers={"Accept-Language": "ar-EG,ar;q=0.9"})
    assert page.status_code == 200
    html = page.text
    # Every occurrence of the label is entity-escaped text — no tag ever reaches the DOM.
    assert "<script>alert(1)" not in html and "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html and "&lt;img src=x onerror=alert(2)&gt;" in html
    assert 'dir="rtl"' in html and 'lang="ar"' in html
    assert 'property="og:image"' in html and "/api/link-preview.png?lat=30.0444&amp;lng=31.2357" in html
    assert f"routemind://v1/p/{lid}" in html
    assert "intent://v1/p/" in html and "package=com.routemind.app" in html
    assert "maps.google.com" not in html  # coordinates line, not a competitor-only page
    assert "https://www.google.com/maps/search/?api=1&amp;query=30.0444%2C31.2357" in html
    assert "https://maps.apple.com/?ll=30.0444%2C31.2357" in html
    assert "https://waze.com/ul?ll=30.0444%2C31.2357&amp;navigate=yes" in html
    assert "app-id=123456789" in html
    assert "content-security-policy" in page.headers and "script-src 'nonce-" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"


def test_landing_page_english_default_and_route(client):
    lid = client.post("/api/links", json={"link": ROUTE}).json()["id"]
    page = client.get(f"/r/{lid}")
    assert page.status_code == 200
    assert 'lang="en"' in page.text and "via 1 stops" in page.text and "Pyramids" in page.text
    assert "https://www.google.com/maps/dir/?api=1&amp;destination=29.9792%2C31.1342" in page.text


def test_expired_and_revoked_pages_still_render(client):
    r = client.post("/api/links", json={"link": PLACE})
    lid, tok = r.json()["id"], r.json()["edit_token"]
    client.post(f"/api/links/{lid}/revoke", headers={"X-Edit-Token": tok})
    page = client.get(f"/p/{lid}")
    assert page.status_code == 410
    assert "30.0459, 31.2243" in page.text and "The sender removed this link" in page.text
    assert client.get("/p/nope1234").status_code == 404


def test_stateless_go_and_open(client):
    page = client.get("/go", params={"lat": "30.0444", "lng": "31.2357", "name": "Cafe"})
    assert page.status_code == 200 and "Cafe" in page.text and "30.0444, 31.2357" in page.text
    assert client.get("/go", params={"lat": "x"}).status_code == 404
    page = client.get("/open", params={"u": "https://maps.app.goo.gl/AbC"})
    assert page.status_code == 200 and "routemind://v1/open?u=https%3A%2F%2Fmaps.app.goo.gl%2FAbC" in page.text
    assert client.get("/open", params={"u": "javascript:alert(1)"}).status_code == 404


def test_preview_png_fallback_without_token(client):
    r = client.get("/api/link-preview.png", params={"lat": "30", "lng": "31"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:8] == b"\x89PNG\r\n\x1a\n"


# ── live trips ────────────────────────────────────────────────────────────────

def test_live_trip_lifecycle(client):
    r = client.post("/api/live", json={"destination": PLACE, "label": "Ahmed", "ttl_minutes": 30})
    assert r.status_code == 201, r.text
    lid, tok = r.json()["id"], r.json()["token"]
    assert r.json()["url"] == f"https://links.test/l/{lid}"

    v = client.get(f"/api/live/{lid}").json()
    assert v["position"] is None and v["ended"] is False and v["label"] == "Ahmed"

    assert client.post(f"/api/live/{lid}/progress", json={"lat": 30.05, "lng": 31.24, "eta_seconds": 600},
                       headers={"X-Live-Token": "bad"}).status_code == 403
    ok = client.post(f"/api/live/{lid}/progress", json={"lat": 30.05, "lng": 31.24, "eta_seconds": 600, "remaining_m": 4200},
                     headers={"X-Live-Token": tok})
    assert ok.status_code == 200
    v = client.get(f"/api/live/{lid}").json()
    assert v["position"]["lat"] == 30.05 and v["eta_seconds"] == 600 and v["position"]["updated_at"]
    assert "token" not in v and "token_hash" not in v

    page = client.get(f"/l/{lid}")
    assert page.status_code == 200 and "Ahmed" in page.text and f"routemind://v1/live/{lid}" in page.text

    assert client.post(f"/api/live/{lid}/end", headers={"X-Live-Token": tok}).status_code == 200
    assert client.get(f"/api/live/{lid}").json()["ended"] is True
    assert client.get("/api/live/nope").status_code == 404
    assert client.get("/l/nope").status_code == 404


def test_live_rejects_destination_without_point(client):
    assert client.post("/api/live", json={"destination": {"kind": "query", "query": "x"}}).status_code == 422


# ── well-known ────────────────────────────────────────────────────────────────

def test_assetlinks(client):
    r = client.get("/.well-known/assetlinks.json")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body[0]["relation"] == ["delegate_permission/common.handle_all_urls"]
    assert body[0]["target"]["package_name"] == "com.routemind.app"
    assert body[0]["target"]["sha256_cert_fingerprints"] == ["AA:BB:CC", "DD:EE:FF"]


def test_aasa_both_paths(client):
    for path in ("/.well-known/apple-app-site-association", "/apple-app-site-association"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
        d = r.json()["applinks"]["details"][0]
        assert d["appIDs"] == ["R3D75N7XCX.com.routemind.app"]
        paths = [c["/"] for c in d["components"]]
        assert "/p/*" in paths and "/l/*" in paths and "/go" in paths
        assert any(c.get("exclude") for c in d["components"] if c["/"] == "/api/*")


# ── resolver: SSRF guards ─────────────────────────────────────────────────────

def test_resolver_rejects_disallowed_targets():
    for bad in ("ftp://maps.google.com/x", "https://evil.example/x", "https://user:pw@maps.google.com/",
                "https://maps.google.com:8443/x", "javascript:alert(1)", "https://169.254.169.254/latest"):
        with pytest.raises(lr.ResolveError):
            lr.sanitise(bad)
    assert lr.sanitise("https://MAPS.APP.GOO.GL/AbC#frag") == "https://maps.app.goo.gl/AbC"


def test_ip_is_public():
    for ip in ("127.0.0.1", "10.1.2.3", "169.254.169.254", "192.168.1.1", "::1", "fd00::1", "::ffff:10.0.0.1", "0.0.0.0"):
        assert not lr.ip_is_public(ip), ip
    assert lr.ip_is_public("142.250.184.206") and lr.ip_is_public("2a00:1450:4001:80b::200e")


def test_resolve_follows_redirects_then_reads_page(monkeypatch):
    calls = []

    async def fake_dns(host):
        calls.append(host)

    monkeypatch.setattr(lr, "check_dns", fake_dns)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "maps.app.goo.gl":
            return httpx.Response(302, headers={"location": "https://www.google.com/maps/place/Cafe/data=!4m2!3m1!1s0x1:0x2"})
        if req.url.host == "www.google.com":
            return httpx.Response(200, headers={"content-type": "text/html"},
                                  text='<html><meta property="og:title" content="Cafe - Google Maps"><meta property="og:image" content="https://maps.googleapis.com/maps/api/staticmap?center=30.0459%2C31.2243"></html>')
        return httpx.Response(404)

    real_client = httpx.AsyncClient

    class Patched(real_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr(lr.httpx, "AsyncClient", Patched)
    out = asyncio.run(lr.resolve("https://maps.app.goo.gl/AbC"))
    assert out["hops"] == 1
    assert out["final_url"].startswith("https://www.google.com/maps/place/Cafe/")
    assert out["link"]["point"] == {"lat": 30.0459, "lng": 31.2243} and out["link"]["label"] == "Cafe"
    assert calls == ["maps.app.goo.gl", "www.google.com"]


def test_resolve_refuses_redirect_to_unlisted_host(monkeypatch):
    async def fake_dns(host):
        return None
    monkeypatch.setattr(lr, "check_dns", fake_dns)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})

    class Patched(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)
    monkeypatch.setattr(lr.httpx, "AsyncClient", Patched)
    with pytest.raises(lr.ResolveError) as e:
        asyncio.run(lr.resolve("https://bit.ly/x"))
    assert e.value.code == "host_not_allowed"


def test_resolve_endpoint_maps_errors(client, monkeypatch):
    r = client.post("/api/links/resolve", json={"url": "https://evil.example/x"})
    assert r.status_code == 400 and r.json()["error"] == "host_not_allowed"

    async def boom(url):
        raise lr.ResolveError("timeout")
    monkeypatch.setattr("api.links.resolve_external", boom)
    assert client.post("/api/links/resolve", json={"url": "https://maps.app.goo.gl/x"}).status_code == 504


def test_rate_limit_kicks_in(client):
    from api.links import _mint_bucket
    _mint_bucket.state.clear()
    codes = [client.post("/api/links", json={"link": PLACE}).status_code for _ in range(25)]
    assert 429 in codes and codes[0] == 201
