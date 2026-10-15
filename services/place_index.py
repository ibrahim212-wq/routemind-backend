"""
services/place_index.py — the server's copy of RouteMind's own place index: what the phone's PlaceStore
(android/app/src/main/java/com/routemind/app/places/PlaceStore.kt in the app repo) does, for the backend.

The index is gzip SQLite tiles of Egyptian open data (OpenStreetMap, Overture, … — ODbL), published as GitHub
Release assets of ibrahim212-wq/routemind-places, searched with the app's own engine (services/place_engine).

  manifest   PLACES_MANIFEST_URL (default DEFAULT_MANIFEST_URL) — fetched at most once every CHECK_EVERY_S; a
             failed fetch keeps the last one seen (it is kept on disk, so a restart answers from it) and is
             retried after RETRY_S. Files live at manifest["base"] (else the manifest URL's directory) + file.
  files      PLACE_INDEX_DIR (default /tmp/routemind-places)/<id>.sqlite: the national file (every area, every
             prominent place, every branch of a chain) always, then the tiles within AROUND_M of each query
             point, within DEST_M of a destination, and along a route (rings of ROUTE_RING_M every ROUTE_STEP_M
             over the first ROUTE_AHEAD_M). Each download is checked against the manifest's sha256 — of the
             .gz bytes, as tools/place_index/tiles.py writes it and PlaceStore.kt checks it — inflated, and
             swapped in atomically. state.json records which build each file came from; manifest.json in the
             directory is a FILTERED copy naming only the files present, so engine.Index opens exactly those.
  waiting    a query waits up to WAIT_S for the files it needs (national first, then nearest first), then
             answers from what is here; the rest keep downloading in the background, MAX_DOWNLOADS at a time.
  threads    every sqlite3 connection and every engine call lives on ONE dedicated thread (sqlite3
             connections are thread-bound); downloads run on a small pool; the async API wraps both.

The data is names, categories, addresses, phones, websites and (rarely) OSM opening_hours — never ratings,
reviews, prices or an open-now state, and nothing here invents them (those keys are always None).

Never raises to its callers: no manifest (the release may not exist yet), no network or a bad file → fewer or
no results.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

import httpx

from services.place_engine import engine as _engine

logger = logging.getLogger("routemind.place_index")

DEFAULT_MANIFEST_URL = "https://github.com/ibrahim212-wq/routemind-places/releases/latest/download/manifest.json"
DEFAULT_DIR = "/tmp/routemind-places"
LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(_engine.__file__)), "egypt_lexicon.json")

CHECK_EVERY_S = 24 * 3600        # manifest refresh
RETRY_S = 300                    # after a failed manifest fetch / a failed file, wait this long to retry it
AROUND_M = 6_000.0               # tiles fetched around each query point (the national file covers what is far)
# Cloud Run's /tmp is memory: the cache is capped (raw SQLite bytes) and the least recently wanted tiles go first
# (never the national file, never a tile wanted in the last minute). Downtown Cairo's 6 km is ≈ 25 tiles / 40 MB.
CACHE_CAP_BYTES = int(float(os.getenv("PLACE_INDEX_CAP_MB", "160")) * 1e6)
DEST_M = 3_000.0                 # … around a destination
ROUTE_RING_M = 1_500.0           # … along a route: rings of this radius
ROUTE_STEP_M = 3_000.0           #     every this many metres
ROUTE_AHEAD_M = 60_000.0         #     over the first this many metres
MAX_DOWNLOADS = 4                # concurrent file downloads
WAIT_S = float(os.getenv("PLACE_INDEX_WAIT_S", "8"))      # how long one query waits for its files
MANIFEST_WAIT_S = 6.0            # how long a query waits for the very first manifest (it keeps loading after)
HTTP_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_UA = {"User-Agent": "RouteMind-backend/1.0 (place-index)"}

UPSTREAM_FILE = "manifest.upstream.json"   # the manifest as published (kept for restarts)
FILTERED_FILE = "manifest.json"            # what engine.Index reads: only the files present
STATE_FILE = "state.json"                  # id → the manifest entry of the file on disk

# One word per category that the engine parses as exactly that category (the lexicon's own terms). Callers map
# their vocabulary onto these: the app's stop categories (fuel, …) and the Google place types the copilot's
# natural-language resolver still produces (gas_station, …).
CATEGORY_WORDS: Dict[str, str] = {
    "gas_station": "بنزينة", "fuel": "بنزينة",
    "pharmacy": "صيدلية",
    "atm": "atm",
    "restaurant": "مطعم",
    "cafe": "كافيه",
    "parking": "جراج",
    "hospital": "مستشفى",
    "mosque": "مسجد",
}

Fetch = Callable[[str], Optional[bytes]]
LatLng = Tuple[float, float]


def http_get(url: str) -> Optional[bytes]:
    """The default fetcher: the body on HTTP 200 (redirects followed — release assets redirect to a CDN), a
    local file for file:// URLs, None on any failure. Runs on the download pool, never on the event loop."""
    try:
        if url.startswith("file://"):
            with open(unquote(urlparse(url).path), "rb") as f:
                return f.read()
        with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, headers=_UA) as c:
            r = c.get(url)
            if r.status_code != 200:
                logger.warning(f"place index: GET {url} → HTTP {r.status_code}")
                return None
            return r.content
    except Exception as e:
        logger.warning(f"place index: GET {url} failed: {e}")
        return None


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    return _engine.haversine(lat1, lng1, lat2, lng2)


_ID_RX = re.compile(r"^[A-Za-z0-9_-]{1,64}$")   # ids name files in the cache dir: no paths


def _valid_entry(e) -> bool:
    return (isinstance(e, dict) and isinstance(e.get("id"), str) and bool(_ID_RX.match(e["id"]))
            and isinstance(e.get("file"), str)
            and isinstance(e.get("sha256"), str) and isinstance(e.get("bbox"), list) and len(e["bbox"]) == 4)


def _parse_manifest(raw: bytes) -> Optional[dict]:
    try:
        m = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(m, dict) or not _valid_entry(m.get("national")) or not isinstance(m.get("tiles"), list):
        return None
    m["tiles"] = [t for t in m["tiles"] if _valid_entry(t)]
    return m


def _write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _tiles_near(m: dict, lat: float, lng: float, radius_m: float) -> List[dict]:
    """The manifest's tiles whose box comes within radius_m of the point, nearest first (engine.Index.near)."""
    dlat = radius_m / 111000.0
    dlng = radius_m / (111000.0 * max(0.2, math.cos(math.radians(lat))))
    out = []
    for t in m["tiles"]:
        x0, y0, x1, y1 = t["bbox"]
        if x1 >= lng - dlng and x0 <= lng + dlng and y1 >= lat - dlat and y0 <= lat + dlat:
            out.append((_haversine(lat, lng, min(max(lat, y0), y1), min(max(lng, x0), x1)), t["id"], t))
    out.sort(key=lambda x: (x[0], x[1]))
    return [t for _, _, t in out]


def _route_rings(route: Sequence[LatLng]) -> List[LatLng]:
    """Ring centres every ROUTE_STEP_M along the first ROUTE_AHEAD_M of the route (PlaceStore.ensureRouteLocked),
    walking long segments too, so a sparse route still gets a ring every step."""
    out: List[LatLng] = []
    if not route:
        return out
    out.append(route[0])
    along, next_at = 0.0, ROUTE_STEP_M
    for a, b in zip(route, route[1:]):
        seg = _haversine(a[0], a[1], b[0], b[1])
        while seg > 0 and next_at <= along + seg and next_at <= ROUTE_AHEAD_M:
            f = (next_at - along) / seg
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            next_at += ROUTE_STEP_M
        along += seg
        if along > ROUTE_AHEAD_M:
            break
    return out


def _densify(route: Sequence[LatLng], step_m: float = 400.0) -> List[LatLng]:
    """The route with a vertex at least every step_m: the engine measures a place's distance to the route
    through its vertices, so a sparse route (two points 10 km apart) would read a place on it as kilometres off."""
    if len(route) < 2:
        return list(route)
    out: List[LatLng] = [route[0]]
    for a, b in zip(route, route[1:]):
        n = int(_haversine(a[0], a[1], b[0], b[1]) // step_m)
        for k in range(1, n + 1):
            f = k / (n + 1)
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
        out.append(b)
    return out


class PlaceIndexStore:
    """The lazy, disk-cached place index. One per process (see store()); `fetch` is injectable for tests."""

    def __init__(self, cache_dir: Optional[str] = None, manifest_url: Optional[str] = None,
                 fetch: Optional[Fetch] = None, wait_s: Optional[float] = None):
        self.dir = cache_dir or os.getenv("PLACE_INDEX_DIR") or DEFAULT_DIR
        self.manifest_url = manifest_url or os.getenv("PLACES_MANIFEST_URL") or DEFAULT_MANIFEST_URL
        self.fetch: Fetch = fetch or http_get
        self.wait_s = WAIT_S if wait_s is None else float(wait_s)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="place-index")
        self._net = ThreadPoolExecutor(max_workers=MAX_DOWNLOADS, thread_name_prefix="place-fetch")
        # shared state — guarded by _lock (the event loop, the download pool and the worker all touch it)
        self._lock = threading.Lock()
        self._loaded = False
        self._manifest: Optional[dict] = None
        self._base = ""
        self._checked = 0.0                       # last successful manifest fetch
        self._tried = 0.0                         # last manifest attempt
        self._manifest_fut: Optional[Future] = None
        self._have: Dict[str, dict] = {}          # id → manifest entry of the file on disk
        self._inflight: Dict[str, Future] = {}    # id → its download
        self._failed: Dict[str, float] = {}       # sha256 → when its download last failed
        self._used: Dict[str, float] = {}         # id → when a query last wanted it (eviction order)
        self._gen = 0                             # bumped whenever the searchable set changes
        # worker-thread state
        self._built = -1
        self._index: Optional[_engine.Index] = None
        self._eng: Optional[_engine.Engine] = None
        self._lexicon: Optional[_engine.Lexicon] = None

    # ── async API ───────────────────────────────────────────────────────────

    async def find(self, query: str, lat: float, lng: float, limit: int = 10,
                   route: Optional[Sequence[Sequence[float]]] = None, dest: Optional[Sequence[float]] = None,
                   kind_word: Optional[str] = None) -> List[dict]:
        """Search the index around (lat, lng) — along `route` [(lat, lng), …] / near `dest` (lat, lng) when given.
        kind_word (a CATEGORY_WORDS value) makes it a search of that category: alone when the query is empty,
        else «<word> <query>» (unless the query already names a category or brand) keeping only places of
        that category. [] on any failure."""
        try:
            q = (query or "").strip()
            if not q and not kind_word:
                return []
            lat, lng = float(lat), float(lng)
            rt = _densify([(float(p[0]), float(p[1])) for p in route]) if route else None
            dst = (float(dest[0]), float(dest[1])) if dest else None
            await self._ensure(lat, lng, rt, dst)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._worker, self._find_sync, q, lat, lng,
                                              max(1, int(limit or 1)), rt, dst, kind_word)
        except Exception as e:
            logger.warning(f"place index find({query!r}) failed: {e}")
            return []

    async def by_pid(self, pid: str, lat: Optional[float] = None, lng: Optional[float] = None) -> Optional[dict]:
        """One place by its stable id ("rm:<pid>" or the bare pid), nearest file first. None when unknown."""
        try:
            p = (pid or "").strip()
            if p.startswith("rm:"):
                p = p[3:]
            if not p:
                return None
            if lat is not None and lng is not None:
                lat, lng = float(lat), float(lng)
                await self._ensure(lat, lng, None, None)
            else:
                await self._ensure(None, None, None, None)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._worker, self._by_pid_sync, p, lat, lng)
        except Exception as e:
            logger.warning(f"place index by_pid({pid!r}) failed: {e}")
            return None

    def close(self) -> None:
        """Close the connections (on their thread) and stop the pools — tests / shutdown only."""
        try:
            self._worker.submit(self._close_sync).result(timeout=10)
        except Exception:
            pass
        self._worker.shutdown(wait=False)
        self._net.shutdown(wait=False, cancel_futures=True)

    def wait_idle(self, timeout: float = 60.0) -> None:
        """Block until no download is in flight (tests)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                futs = list(self._inflight.values()) + ([self._manifest_fut] if self._manifest_fut else [])
            if not futs:
                return
            for f in futs:
                try:
                    f.result(timeout=max(0.0, end - time.time()))
                except Exception:
                    pass

    # ── files ───────────────────────────────────────────────────────────────

    async def _ensure(self, lat: Optional[float], lng: Optional[float], route, dest) -> None:
        """The manifest (waited for only when there is none yet), then the files this query needs."""
        mfut = self._manifest_due()
        if mfut is not None:
            await asyncio.wait([asyncio.wrap_future(mfut)], timeout=MANIFEST_WAIT_S)
        futs = self._schedule(lat, lng, route, dest)
        if futs and self.wait_s > 0:
            await asyncio.wait([asyncio.wrap_future(f) for f in futs], timeout=self.wait_s)

    def _load_disk_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"place index: cannot create {self.dir}: {e}")
            return
        try:
            up = os.path.join(self.dir, UPSTREAM_FILE)
            if os.path.exists(up):
                with open(up, "rb") as f:
                    m = _parse_manifest(f.read())
                if m is not None:
                    self._manifest, self._base = m, self._base_of(m)
                    self._checked = os.path.getmtime(up)
        except Exception as e:
            logger.warning(f"place index: stored manifest unreadable: {e}")
        try:
            sp = os.path.join(self.dir, STATE_FILE)
            if os.path.exists(sp):
                with open(sp, encoding="utf-8") as f:
                    for tid, e in (json.load(f) or {}).items():
                        if _valid_entry(e) and os.path.exists(os.path.join(self.dir, tid + ".sqlite")):
                            self._have[tid] = e
        except Exception as e:
            logger.warning(f"place index: state unreadable: {e}")
        self._gen += 1

    def _save_state_locked(self) -> None:
        try:
            _write_atomic(os.path.join(self.dir, STATE_FILE),
                          json.dumps(self._have, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            logger.warning(f"place index: state not saved: {e}")

    def _base_of(self, m: dict) -> str:
        base = m.get("base") if isinstance(m.get("base"), str) and m.get("base") else \
            self.manifest_url.rsplit("/", 1)[0] + "/"
        return base if base.endswith("/") else base + "/"

    def _manifest_due(self) -> Optional[Future]:
        """Start a manifest fetch when one is due; the future to wait on when there is no manifest at all."""
        with self._lock:
            self._load_disk_locked()
            now = time.time()
            stale = self._manifest is None or now - self._checked >= CHECK_EVERY_S
            if stale and self._manifest_fut is None and now - self._tried >= RETRY_S:
                self._tried = now
                self._manifest_fut = self._net.submit(self._fetch_manifest)
            return self._manifest_fut if self._manifest is None else None

    def _fetch_manifest(self) -> bool:
        try:
            raw = self.fetch(self.manifest_url)
            m = _parse_manifest(raw) if raw else None
            if m is None:
                logger.warning(f"place index: manifest unavailable at {self.manifest_url} — "
                               f"{'keeping the last one' if self._manifest else 'no index yet'}")
                return False
            try:
                _write_atomic(os.path.join(self.dir, UPSTREAM_FILE), raw)
            except Exception as e:
                logger.warning(f"place index: manifest not stored: {e}")
            with self._lock:
                old = (self._manifest or {}).get("version")
                self._manifest, self._base, self._checked = m, self._base_of(m), time.time()
                self._failed.clear()
                self._gen += 1
            if old != m.get("version"):
                logger.info(f"place index: manifest {old} → {m.get('version')}")
            return True
        except Exception as e:
            logger.warning(f"place index: manifest fetch failed: {e}")
            return False
        finally:
            with self._lock:
                self._manifest_fut = None

    def _schedule(self, lat, lng, route, dest) -> List[Future]:
        """Start the downloads this query needs (national, around the point, the destination, the route ahead)
        and return every future it should wait for — new or already in flight."""
        with self._lock:
            m = self._manifest
            if m is None:
                return []
            want = [m["national"]]
            if lat is not None and lng is not None:
                want += _tiles_near(m, lat, lng, AROUND_M)
            if dest:
                want += _tiles_near(m, dest[0], dest[1], DEST_M)
            if route:
                for p in _route_rings(route):
                    want += _tiles_near(m, p[0], p[1], ROUTE_RING_M)
            now = time.time()
            futs: List[Future] = []
            seen = set()
            for e in want:
                tid, sha = e["id"], e["sha256"]
                if tid in seen:
                    continue
                seen.add(tid)
                self._used[tid] = now
                if (self._have.get(tid) or {}).get("sha256") == sha:
                    continue
                f = self._inflight.get(tid)
                if f is None:
                    if now - self._failed.get(sha, 0.0) < RETRY_S:
                        continue
                    f = self._net.submit(self._download, dict(e), self._base)
                    self._inflight[tid] = f
                futs.append(f)
            return futs

    def _download(self, e: dict, base: str) -> bool:
        tid, sha = e["id"], e["sha256"]
        ok = False
        tmp = os.path.join(self.dir, f".{tid}.{sha[:12]}.part")
        try:
            gz = self.fetch(base + e["file"])
            if gz is None:
                return False
            if hashlib.sha256(gz).hexdigest() != sha:
                logger.warning(f"place index: {tid} sha256 mismatch — dropped")
                return False
            with gzip.GzipFile(fileobj=io.BytesIO(gz)) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 16)
            os.replace(tmp, os.path.join(self.dir, tid + ".sqlite"))
            ok = True
            return True
        except Exception as ex:
            logger.warning(f"place index: {tid} download failed: {ex}")
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False
        finally:
            with self._lock:
                self._inflight.pop(tid, None)
                if ok:
                    self._have[tid] = e
                    self._gen += 1
                    self._evict_locked()
                    self._save_state_locked()
                else:
                    self._failed[sha] = time.time()

    def _evict_locked(self) -> None:
        """Keep the cache under CACHE_CAP_BYTES. A deleted file stays readable by a connection that has it
        open (POSIX) until the engine is rebuilt without it (_gen)."""
        total = sum(int(e.get("raw_bytes") or 0) for e in self._have.values())
        if total <= CACHE_CAP_BYTES:
            return
        national = ((self._manifest or {}).get("national") or {}).get("id", "national")
        now = time.time()
        for tid in sorted((t for t in self._have if t != national), key=lambda t: (self._used.get(t, 0.0), t)):
            if total <= CACHE_CAP_BYTES:
                break
            if now - self._used.get(tid, 0.0) < 60:
                continue
            e = self._have.pop(tid)
            total -= int(e.get("raw_bytes") or 0)
            try:
                os.remove(os.path.join(self.dir, tid + ".sqlite"))
            except OSError:
                pass
            self._gen += 1

    def warm(self) -> None:
        """Start the manifest + national file now (called at startup) so the first copilot answer is fast."""
        try:
            fut = self._manifest_due()
            if fut is not None:
                fut.add_done_callback(lambda f: self._schedule(None, None, None, None))
            else:
                self._schedule(None, None, None, None)
        except Exception as e:  # never at the cost of startup
            logger.warning(f"place index: warm-up failed: {e}")

    # ── worker thread: the index, the engine, the rows ──────────────────────

    def _close_sync(self) -> None:
        if self._index is not None:
            for con in self._index.cons.values():
                try:
                    con.close()
                except Exception:
                    pass
        self._index = self._eng = None

    def _engine_now(self) -> Optional[_engine.Engine]:
        """The engine over the files present now — rebuilt when files landed or the manifest changed."""
        with self._lock:
            gen, have, m = self._gen, dict(self._have), self._manifest
        if gen == self._built:
            return self._eng
        if self._lexicon is None:
            self._lexicon = _engine.Lexicon.load(LEXICON_PATH)
        self._close_sync()
        current = None if m is None else {m["national"]["id"]} | {t["id"] for t in m["tiles"]}
        stale = [tid for tid in have if current is not None and tid not in current]
        if stale:  # a newer build dropped these boxes: forget them
            with self._lock:
                for tid in stale:
                    self._have.pop(tid, None)
                self._save_state_locked()
            for tid in stale:
                have.pop(tid, None)
                try:
                    os.remove(os.path.join(self.dir, tid + ".sqlite"))
                except OSError:
                    pass
        self._built = gen
        if not have:
            return None
        meta = {k: v for k, v in (m or {}).items() if k not in ("national", "tiles", "totals")}
        national = have.get("national") or (m or {}).get("national") or \
            {"id": "national", "bbox": [0, 0, 0, 0], "file": "", "sha256": "", "places": 0}
        filtered = dict(meta, national=dict(national, places=national.get("places", 0)),
                        tiles=[dict(e, places=e.get("places", 0)) for tid, e in sorted(have.items())
                               if tid != "national"])
        _write_atomic(os.path.join(self.dir, FILTERED_FILE), json.dumps(filtered, ensure_ascii=False).encode("utf-8"))
        self._index = _engine.Index(self.dir)
        if not self._index.cons:
            self._close_sync()
            return None
        self._eng = _engine.Engine(self._index, self._lexicon)
        return self._eng

    def _find_sync(self, q: str, lat: float, lng: float, limit: int, route, dest, kind_word) -> List[dict]:
        eng = self._engine_now()
        if eng is None:
            return []
        family = None
        if kind_word:
            if not q:
                q = kind_word
            else:
                want = eng.parse(kind_word, lat, lng).scat
                family = _engine.SCAT_FAMILY.get(want, (want,)) if want else None
                p = eng.parse(q, lat, lng)
                if not (p.scat or p.brand):
                    q = f"{kind_word} {q}"
        # a typed name search keeps only that category («بنزينة Master» → fuel stations named Master, not
        # "Master Gym"): the engine ranks a strong name match of another kind high, so over-fetch, then filter
        rs = eng.search(q, lat, lng, limit=limit if family is None else max(3 * limit, 20), route=route, dest=dest)
        if family is not None:
            rs = [r for r in rs if r.scat in family][:limit]
        out = []
        for r in rs:
            d = self._row_dict(r.db, r.row or {}, lat, lng)
            if d is not None:
                out.append(d)
        return out

    def _by_pid_sync(self, pid: str, lat, lng) -> Optional[dict]:
        eng = self._engine_now()
        if pid.startswith("lex:"):
            return self._lexicon_row(pid, lat, lng)
        if eng is None or self._index is None:
            return None
        ix = self._index
        near = ix.near(lat, lng, 2000.0) if lat is not None and lng is not None else []
        for db in dict.fromkeys(near + ["national"] + sorted(ix.cons)):
            con = ix.cons.get(db)
            if con is None:
                continue
            row = con.execute("SELECT * FROM place WHERE pid = ? LIMIT 1", (pid,)).fetchone()
            if row is not None:
                return self._row_dict(db, dict(row), lat, lng)
        return None

    def _lexicon_row(self, pid: str, lat, lng) -> Optional[dict]:
        """A lexicon pin («برج القاهرة» answered from the lexicon itself) — the engine's own lex:<id> results."""
        if self._lexicon is None:
            self._lexicon = _engine.Lexicon.load(LEXICON_PATH)
        want = pid[4:]
        for ent in self._lexicon.entities.values():
            if ent.get("id") == want:
                cat = {"area": "neighbourhood", "road": "road", "landmark": "landmark"}.get(ent.get("kind"), "landmark")
                return self._row_dict("lexicon", {"pid": pid, "name_ar": ent.get("ar"), "name_en": ent.get("en"),
                                                  "lat": ent["lat"], "lng": ent["lng"], "cat": cat, "scat": None},
                                      lat, lng)
        return None

    def _area_name(self, db: str, area_id, arabic: bool) -> Optional[str]:
        if area_id is None or self._index is None or db not in self._index.cons:
            return None
        try:
            r = self._index.cons[db].execute("SELECT ar, en FROM area WHERE id = ?", (area_id,)).fetchone()
        except Exception:
            return None
        if r is None:
            return None
        return (r["ar"] or r["en"]) if arabic else (r["en"] or r["ar"])

    def _row_dict(self, db: str, row: dict, lat, lng) -> Optional[dict]:
        """An index row → the backend's place dict. Ratings, reviews, price and open-now do not exist in open
        data: always None, never guessed. hours is the raw OSM opening_hours string when the source has one."""
        plat, plng = row.get("lat"), row.get("lng")
        if plat is None or plng is None or not row.get("pid"):
            return None
        name_ar, name_en = row.get("name_ar") or None, row.get("name_en") or None
        scat = row.get("scat") or row.get("cat")
        name = name_ar or name_en
        if not name and self._lexicon is not None:
            name = (self._lexicon.labels.get(scat) or (None, None))[0]
        if not name:
            return None
        addr = (row.get("addr") or "").strip() or None
        area = self._area_name(db, row.get("area"), arabic=bool(name_ar))
        if area and addr and area in addr:
            area = None
        address = ", ".join(x for x in (addr, area) if x) or None
        return {
            "id": "rm:" + row["pid"],
            "name": name,
            "name_ar": name_ar,
            "name_en": name_en,
            "lat": float(plat),
            "lng": float(plng),
            "address": address,
            "category": scat,
            "distance_m": int(round(_haversine(lat, lng, plat, plng))) if lat is not None and lng is not None else None,
            "phone": row.get("phone") or None,
            "website": row.get("website") or None,
            "hours": row.get("hours") or None,
            "rating": None,
            "reviews": None,
            "price": None,
            "open_now": None,
        }


# ── the process-wide store ─────────────────────────────────────────────────────

_store: Optional[PlaceIndexStore] = None
_store_lock = threading.Lock()


def store() -> PlaceIndexStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = PlaceIndexStore()
        return _store


def configure(**kwargs) -> PlaceIndexStore:
    """Replace the process-wide store (tests: another directory, a file:// manifest, a stub fetcher)."""
    global _store
    with _store_lock:
        old, _store = _store, PlaceIndexStore(**kwargs)
    if old is not None:
        old.close()
    return _store


async def find(query: str, lat: float, lng: float, limit: int = 10, route=None, dest=None,
               kind_word: Optional[str] = None) -> List[dict]:
    return await store().find(query, lat, lng, limit=limit, route=route, dest=dest, kind_word=kind_word)


async def by_pid(pid: str, lat: Optional[float] = None, lng: Optional[float] = None) -> Optional[dict]:
    return await store().by_pid(pid, lat, lng)
