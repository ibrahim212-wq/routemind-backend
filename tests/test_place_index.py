"""services/place_index.py and the api/places.py paths on it — against a local copy of the place index, no network.

The store reads a file:// manifest (the default fetcher reads file:// URLs; the stub below refuses anything else)
into a temp cache dir, exactly as it would read the GitHub release: sha256 check, gunzip, filtered manifest.
The index-backed tests need a built index (the app repo's `python -m tools.place_index.build --out <dir>`):
ROUTEMIND_PLACES_TEST_INDEX names that directory (manifest.json + <id>.sqlite.gz); without one they skip.
The degradation tests (no manifest, bad files) always run.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from api import places
from services import place_index
from services import route_geometry as geo

IDX = os.getenv("ROUTEMIND_PLACES_TEST_INDEX", "")
HAVE_INDEX = bool(IDX) and os.path.exists(os.path.join(IDX, "manifest.json"))
needs_index = pytest.mark.skipif(not HAVE_INDEX, reason=f"no built place index at {IDX}")

MAADI = (29.9602, 31.2569)
TAHRIR = (30.0444, 31.2357)
# Maadi → Tahrir up the Nile corniche, (lat, lng)
ROUTE = [MAADI, (29.9700, 31.2470), (29.9860, 31.2330), (30.0050, 31.2300), (30.0200, 31.2310),
         (30.0330, 31.2320), TAHRIR]


def run(coro):
    return asyncio.run(coro)


def _file_only(url: str):
    assert url.startswith("file://"), f"network fetch attempted in a test: {url}"
    return place_index.http_get(url)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """The process-wide store pointed at the local index (api/places.py searches through it)."""
    old = place_index._store
    s = place_index.PlaceIndexStore(cache_dir=str(tmp_path_factory.mktemp("places")),
                                    manifest_url="file://" + os.path.join(IDX, "manifest.json"),
                                    fetch=_file_only, wait_s=120)
    place_index._store = s
    yield s
    s.wait_idle()
    s.close()
    place_index._store = old


@pytest.fixture
def no_overpass(monkeypatch):
    calls = []

    async def fake(category, bbox):
        calls.append(category)
        return []
    monkeypatch.setattr(places.overpass, "fetch_pois", fake)
    return calls


# ── against the index ─────────────────────────────────────────────────────────

@needs_index
def test_category_near_maadi_returns_gas_stations(store):
    rs = run(place_index.find("", *MAADI, limit=5, kind_word=place_index.CATEGORY_WORDS["gas_station"]))
    assert len(rs) >= 3
    assert all(r["category"] == "gas_station" for r in rs)
    assert min(r["distance_m"] for r in rs) < 3000
    for r in rs:
        assert r["id"].startswith("rm:") and r["name"]
        assert r["rating"] is None and r["reviews"] is None and r["price"] is None and r["open_now"] is None
    # the cache holds the national file + the tiles around the point, and a filtered manifest naming them
    files = set(os.listdir(store.dir))
    assert "national.sqlite" in files and "manifest.json" in files and "state.json" in files
    # the API resolver: nearest first, as PlaceResults
    near = run(places.resolve_nearby(*MAADI, category="gas_station", limit=3))
    assert len(near) == 3
    assert [p.distance_m for p in near] == sorted(p.distance_m for p in near)
    assert near[0].distance_m == min(r["distance_m"] for r in rs)


@needs_index
def test_name_query_finds_as_salam_hospital(store):
    rs = run(place_index.find("مستشفى السلام", *MAADI, limit=5))
    hits = [r for r in rs if "السلام" in (r["name_ar"] or "") and r["category"] == "hospital"]
    assert hits, [r["name"] for r in rs]
    text = run(places.search_places(lat=MAADI[0], lng=MAADI[1], query="مستشفى السلام", limit=5))
    assert any("السلام" in (r["name_ar"] or "") for r in text)
    # a place type constrains a name search to that category (the old Google includedType)
    typed = run(places.search_places(lat=MAADI[0], lng=MAADI[1], query="Master", included_type="gas_station"))
    assert all(r["category"] == "gas_station" for r in typed)
    kfc = run(place_index.find("kfc", *MAADI, limit=3, kind_word=place_index.CATEGORY_WORDS["restaurant"]))
    assert kfc and all(r["category"] in ("restaurant", "fast_food", "burger") for r in kfc)


@needs_index
def test_along_route_fuel_between_maadi_and_tahrir(store, no_overpass):
    req = places.AlongRoutePoisRequest(
        category="fuel", current_location=places.LatLng(lat=MAADI[0], lng=MAADI[1]),
        route=[[lng, lat] for lat, lng in ROUTE], max_results=10)
    out = run(places.along_route_pois(req))
    assert out.source == "index" and out.results, out
    assert no_overpass == []
    along = [r.along_route_distance_m for r in out.results]
    assert along == sorted(along)
    assert all(r.distance_from_route_m <= places.CORRIDOR_WIDTH_M for r in out.results)
    assert all(r.category == "fuel" and r.id.startswith("rm:") for r in out.results)
    # every hit really is a fuel station in the index
    for r in out.results[:3]:
        assert run(place_index.by_pid(r.id))["category"] == "gas_station"
    # the copilot's along-route text search goes through the same engine with the route
    poly = geo.encode_polyline5(ROUTE)
    rs = run(places.search_places(lat=MAADI[0], lng=MAADI[1], query="بنزينة", route_polyline=poly, limit=5))
    assert rs and all(r["category"] == "gas_station" for r in rs)


@needs_index
@pytest.mark.parametrize("category", ["fuel", "restaurant", "cafe", "atm", "parking"])
def test_the_five_stop_categories_along_the_route(store, no_overpass, category):
    remaining = geo.remaining_route(ROUTE, MAADI)
    cum = geo.cumulative_distances(remaining)
    raw = run(places._google_corridor_pois(category, remaining))
    pois = places._filter_to_corridor(raw, remaining, cum, category, 10)
    want = place_index.CATEGORY_WORDS[category]
    assert want
    if category != "parking":       # downtown Cairo has every one of these along the corniche
        assert pois, category
    assert all(p.distance_from_route_m <= places.CORRIDOR_WIDTH_M for p in pois)


@needs_index
def test_by_pid_round_trip(store):
    r = run(place_index.find("", *MAADI, limit=1, kind_word="صيدلية"))[0]
    assert run(place_index.by_pid(r["id"], *MAADI)) == r
    bare = run(place_index.by_pid(r["id"][3:]))
    assert bare == dict(r, distance_m=None)
    det = run(places.place_details_rich(r["id"]))
    assert det["name"] == r["name"] and det["review_samples"] == [] and det["editorial"] is None
    assert det["rating"] is None and (det["hours"] is None or det["hours"] == [r["hours"]])
    page = run(places.search_place_details(r["id"]))
    assert page.name == r["name"] and page.rating is None
    sug = run(places.search_autocomplete(q=r["name"], lat=MAADI[0], lng=MAADI[1]))
    assert sug and all(x.place_id.startswith("rm:") and x.primary_text for x in sug)
    assert run(place_index.by_pid("rm:no-such-place")) is None


# ── degradation: never raise, answer empty ────────────────────────────────────

def test_unreachable_manifest_answers_empty(tmp_path, monkeypatch):
    calls = []

    def down(url):
        calls.append(url)
        return None            # what http_get returns for a 404 / timeout / DNS failure
    s = place_index.PlaceIndexStore(cache_dir=str(tmp_path), manifest_url="https://example.invalid/manifest.json",
                                    fetch=down, wait_s=1)
    try:
        assert run(s.find("بنزينة", *MAADI)) == []
        assert run(s.find("", *MAADI, kind_word="مطعم", route=ROUTE)) == []
        assert run(s.by_pid("rm:anything", *MAADI)) is None
        assert calls == ["https://example.invalid/manifest.json"]   # retried only after RETRY_S, not per query
        monkeypatch.setattr(place_index, "_store", s)
        assert run(places.resolve_nearby(*MAADI, category="pharmacy")) == []
        assert run(places.search_places(lat=MAADI[0], lng=MAADI[1], query="كافيه")) == []
        assert run(places.place_details_rich("rm:x")) is None
        assert run(places.search_autocomplete(q="mall")) == []
    finally:
        s.close()


def test_missing_manifest_file_answers_empty(tmp_path):
    s = place_index.PlaceIndexStore(cache_dir=str(tmp_path / "cache"),
                                    manifest_url="file://" + str(tmp_path / "nope" / "manifest.json"), wait_s=1)
    try:
        assert run(s.find("صيدلية", *TAHRIR)) == []
    finally:
        s.close()


@needs_index
def test_files_failing_the_sha256_check_are_dropped(tmp_path):
    def tampered(url):
        data = _file_only(url)
        return data if url.endswith("manifest.json") else b"\x1f\x8b not the published file"
    s = place_index.PlaceIndexStore(cache_dir=str(tmp_path), fetch=tampered, wait_s=30,
                                    manifest_url="file://" + os.path.join(IDX, "manifest.json"))
    try:
        assert run(s.find("بنزينة", *MAADI)) == []
        assert not [f for f in os.listdir(tmp_path) if f.endswith(".sqlite")]
    finally:
        s.wait_idle()
        s.close()


def test_polyline5_round_trip():
    pts = [MAADI, (29.99, 31.23), TAHRIR, (-12.5, -70.12346)]
    back = places._decode_polyline5(geo.encode_polyline5(pts))
    assert len(back) == len(pts)
    assert all(abs(a - c) < 1e-5 and abs(b - d) < 1e-5 for (a, b), (c, d) in zip(pts, back))


@needs_index
def test_the_cache_stays_under_its_cap_and_keeps_the_national_file(tmp_path, monkeypatch):
    """Cloud Run's /tmp is memory: past the cap the least recently wanted tiles go — never the national file."""
    monkeypatch.setattr(place_index, "CACHE_CAP_BYTES", 30_000_000)
    s = place_index.PlaceIndexStore(cache_dir=str(tmp_path), fetch=_file_only, wait_s=120,
                                    manifest_url="file://" + os.path.join(IDX, "manifest.json"))
    try:
        for (lat, lng) in (TAHRIR, (31.2156, 29.9553), (24.0889, 32.8998)):   # Cairo, Alexandria, Aswan
            run(s.find("صيدلية", lat, lng, limit=3))
            s.wait_idle()
            for tid in list(s._used):                                       # age every wanted tile
                s._used[tid] -= 120
        have = s._have
        raw = sum(int(e.get("raw_bytes") or 0) for e in have.values())
        national = s._manifest["national"]["id"]
        assert national in have
        assert raw <= 30_000_000 + max(int(e.get("raw_bytes") or 0) for e in have.values())
        on_disk = {f[:-7] for f in os.listdir(tmp_path) if f.endswith(".sqlite")}
        assert on_disk == set(have)
        assert run(s.find("صيدلية", 24.0889, 32.8998, limit=3))   # the last city still answers
    finally:
        s.wait_idle()
        s.close()
