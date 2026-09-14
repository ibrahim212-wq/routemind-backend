"""
services/link_resolver.py — server-side resolution of THIRD-PARTY short links.

Why on the server: maps.app.goo.gl / goo.gl / waze hsv / maps.apple short
links redirect through consent and interstitial pages that differ by
region, and the final Google place page carries no @lat,lng for a share
(only an ftid); the coordinates live in the page's og:image `center=` and
`!3d!4d` blobs. The phone can follow redirects too (the Dart client does
when this endpoint is unreachable), but doing it here keeps the client
simple and lets us cache.

SSRF hardening (docs/deep-links-research.md §8):
  • strict host ALLOWLIST (map services and shorteners only), checked on the
    initial URL AND on every redirect hop;
  • http(s) only; credentials in the URL rejected; fragments dropped;
  • DNS resolved and every address checked against private / loopback /
    link-local / metadata / IPv4-mapped ranges before connecting
    (re-resolved per hop — a hop can point back inside);
  • redirects followed manually, at most MAX_HOPS;
  • per-hop timeout, total deadline, response size cap, only text/html read;
  • no HEAD-then-GET games: GET with a small cap is one request.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from services.link_grammar import extract_from_html, valid_point

logger = logging.getLogger("routemind.links.resolver")

MAX_HOPS = 6
HOP_TIMEOUT = 6.0
TOTAL_TIMEOUT = 14.0
MAX_BODY = 600 * 1024

ALLOWED_HOSTS = {
    # google
    "maps.app.goo.gl", "goo.gl", "g.page", "g.co", "maps.google.com", "www.google.com", "google.com",
    "consent.google.com", "maps.googleapis.com",
    # apple
    "maps.apple", "maps.apple.com",
    # waze
    "waze.com", "www.waze.com", "ul.waze.com", "waze.to",
    # osm / others
    "osm.org", "www.openstreetmap.org", "openstreetmap.org", "omaps.app", "share.here.com", "wego.here.com",
    "mapy.cz", "mapy.com", "go.2gis.com", "2gis.com", "petalmaps.com", "www.petalmaps.com", "plus.codes",
    "w3w.co", "what3words.com", "www.what3words.com", "yandex.ru", "yandex.com", "ya.ru", "osmand.net",
    # shorteners people wrap map links in
    "bit.ly", "t.co", "tinyurl.com", "fb.me", "lnkd.in", "cutt.ly", "rb.gy", "is.gd", "ow.ly", "buff.ly",
    "rebrand.ly", "s.id", "shorturl.at", "t.ly", "tiny.cc", "v.gd", "l.facebook.com", "lm.facebook.com",
    "l.instagram.com", "l.messenger.com",
}
ALLOWED_SUFFIXES = (".google.com", ".google.com.eg", ".google.co.uk", ".google.ae", ".google.com.sa",
                    ".waze.com", ".here.com", ".2gis.com", ".2gis.ae", ".yandex.ru", ".yandex.com")

USER_AGENT = ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Mobile Safari/537.36")

_PRIVATE_NETS = [ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
    "192.0.0.0/24", "192.168.0.0/16", "198.18.0.0/15", "224.0.0.0/4", "240.0.0.0/4",
    "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8", "::ffff:0:0/96", "64:ff9b::/96",
)]


class ResolveError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code


def host_allowed(host: str) -> bool:
    h = host.lower().rstrip(".")
    return h in ALLOWED_HOSTS or any(h.endswith(s) for s in ALLOWED_SUFFIXES)


def ip_is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved or addr.is_unspecified:
        return False
    return not any(addr in n for n in _PRIVATE_NETS)


def sanitise(url: str) -> str:
    """Validate scheme/host/credentials; drop the fragment. Raises ResolveError."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise ResolveError("scheme_not_allowed", parts.scheme)
    if not parts.hostname:
        raise ResolveError("no_host")
    if parts.username or parts.password:
        raise ResolveError("credentials_in_url")
    if not host_allowed(parts.hostname):
        raise ResolveError("host_not_allowed", parts.hostname)
    if parts.port not in (None, 80, 443):
        raise ResolveError("port_not_allowed", str(parts.port))
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


async def check_dns(host: str) -> None:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ResolveError("dns_failed", str(e))
    if not infos:
        raise ResolveError("dns_failed", "no addresses")
    for info in infos:
        ip = info[4][0]
        if not ip_is_public(ip):
            raise ResolveError("private_address", ip)


def _continue_url(page: str, base: str) -> Optional[str]:
    """Google's consent interstitial keeps the real target in a `continue` field."""
    if "consent.google" not in base and 'name="continue"' not in page:
        return None
    m = re.search(r'name="continue"\s+value="([^"]+)"', page) or re.search(r'href="(https://www\.google\.com/maps[^"]+)"', page)
    if not m:
        return None
    v = m.group(1).replace("&amp;", "&")
    return v


async def resolve(url: str) -> Dict[str, Any]:
    """
    Follows [url] to its final page. Returns
      {"final_url": str, "hops": int, "link": {...} | None}
    where `link` is a coordinate/place extracted from the final page when the
    final URL itself carries no coordinates (the client re-parses final_url).
    """
    current = sanitise(url)
    deadline = asyncio.get_running_loop().time() + TOTAL_TIMEOUT
    hops = 0
    body = ""
    async with httpx.AsyncClient(follow_redirects=False, timeout=HOP_TIMEOUT,
                                 headers={"User-Agent": USER_AGENT,
                                          "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                                          "Accept-Language": "ar-EG,ar;q=0.9,en;q=0.8"}) as client:
        while True:
            if asyncio.get_running_loop().time() > deadline:
                raise ResolveError("timeout")
            host = urlsplit(current).hostname or ""
            await check_dns(host)
            try:
                async with client.stream("GET", current) as resp:
                    loc = resp.headers.get("location")
                    if 300 <= resp.status_code < 400 and loc:
                        hops += 1
                        if hops > MAX_HOPS:
                            raise ResolveError("too_many_redirects")
                        nxt = urljoin(current, loc)
                        current = sanitise(nxt)
                        continue
                    ctype = resp.headers.get("content-type", "")
                    if resp.status_code >= 400:
                        raise ResolveError("upstream_%d" % resp.status_code)
                    if "html" in ctype or "text" in ctype or not ctype:
                        chunks: List[bytes] = []
                        size = 0
                        async for chunk in resp.aiter_bytes():
                            chunks.append(chunk)
                            size += len(chunk)
                            if size >= MAX_BODY:
                                break
                        body = b"".join(chunks).decode("utf-8", errors="replace")
                    else:
                        body = ""
            except httpx.TimeoutException:
                raise ResolveError("timeout")
            except httpx.HTTPError as e:
                raise ResolveError("fetch_failed", str(e))
            cont = _continue_url(body, current)
            if cont and hops < MAX_HOPS:
                hops += 1
                current = sanitise(cont)
                continue
            break
    link = extract_from_html(body) if body else None
    return {"final_url": current, "hops": hops, "link": link}


def quick_point_in_url(url: str) -> Optional[Tuple[float, float]]:
    """A cheap check the caller may use to stop early (mirrors Dart's parser only loosely)."""
    m = re.search(r"[@?&=/:,](-?\d{1,3}\.\d+)(?:,|%2C)(-?\d{1,3}\.\d+)", url)
    if not m:
        return None
    return valid_point(m.group(1), m.group(2))
