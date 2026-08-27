# -*- coding: utf-8 -*-
"""
tools/check_copilot_parity.py — the CopilotV2 string-catalog parity guard.

Reads the three catalogs — backend api/copilot_strings.py, Android
nav/CopilotStrings.kt, iOS CopilotStrings (inline in
MapboxNavigationViewController.swift) — and fails if:

  • a key exists on one client surface but not the other (Android ↔ iOS must
    be identical; backend keys are a separate, smaller namespace and are
    checked only for internal completeness),
  • any 'ar' value carries no Arabic script, or any 'en' value carries Arabic,
  • a %-style placeholder count differs between the two languages of one key
    (a format crash waiting for the OTHER language to hit it).

Run from the backend repo root:
    python tools/check_copilot_parity.py [--app-root <path-to-cairoWay>]

Exit 0 = parity holds. This is the no-compiler guard for the client catalogs.
"""

import argparse
import re
import sys
from pathlib import Path

AR_RE = re.compile(r"[\u0600-\u06FF]")
# %d, %@, %s, %1$d, %1$@, %.2f …
FMT_RE = re.compile(r"%(?:\d+\$)?[@sdf]|%(?:\d+\$)?\.\d+f")


def fmt_count(s: str) -> int:
    return len(FMT_RE.findall(s))


def parse_kotlin(path: Path):
    src = path.read_text(encoding="utf-8")
    # "key" to Pair(\n? "en", "ar")   — strings may span lines via  + concat
    entries = {}
    pat = re.compile(
        r'"(?P<key>[a-z_0-9]+)"\s+to\s+Pair\(\s*(?P<en>(?:"(?:[^"\\]|\\.)*"\s*\+?\s*)+),'
        r'\s*(?P<ar>(?:"(?:[^"\\]|\\.)*"\s*\+?\s*)+)\)', re.S)
    for m in pat.finditer(src):
        def joined(g):
            return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', g))
        entries[m.group("key")] = (joined(m.group("en")), joined(m.group("ar")))
    return entries


def parse_swift(path: Path):
    src = path.read_text(encoding="utf-8")
    # inside the CopilotStrings enum:  "key": ("en", "ar"),  values may be
    # split across lines with + concatenation
    m = re.search(r"enum CopilotStrings\s*\{.*?static let S:.*?=\s*\[(?P<body>.*?)\n\s*\]",
                  src, re.S)
    if not m:
        return {}
    body = m.group("body")
    entries = {}
    pat = re.compile(
        r'"(?P<key>[a-z_0-9]+)":\s*\(\s*(?P<en>(?:"(?:[^"\\]|\\.)*"\s*\+?\s*)+),'
        r'\s*(?P<ar>(?:"(?:[^"\\]|\\.)*"\s*\+?\s*)+)\)', re.S)
    for mm in pat.finditer(body):
        def joined(g):
            return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', g))
        entries[mm.group("key")] = (joined(mm.group("en")), joined(mm.group("ar")))
    return entries


def parse_backend(root: Path):
    sys.path.insert(0, str(root))
    from api.copilot_strings import _STRINGS  # noqa: E402
    return {k: (v["en"], v["ar"]) for k, v in _STRINGS.items()}


def check_values(name, entries, errors):
    for key, (en, ar) in entries.items():
        if not AR_RE.search(ar):
            errors.append(f"{name}: '{key}' ar value has no Arabic script: {ar!r}")
        if AR_RE.search(en):
            errors.append(f"{name}: '{key}' en value contains Arabic: {en!r}")
        # placeholder parity between the two languages of one key
        if fmt_count(en) != fmt_count(ar):
            errors.append(f"{name}: '{key}' placeholder count differs "
                          f"en={fmt_count(en)} ar={fmt_count(ar)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-root", default=None,
                    help="path to the cairoWay app repo root")
    args = ap.parse_args()

    backend_root = Path(__file__).resolve().parent.parent
    app_root = Path(args.app_root) if args.app_root else None
    if app_root is None:
        for cand in (Path("G:/grad_all/maps-mobile-application/cairoWay"),
                     backend_root.parent / "cairoWay"):
            if cand.exists():
                app_root = cand
                break
    if app_root is None or not app_root.exists():
        print("app repo not found — pass --app-root")
        return 2

    kt = parse_kotlin(app_root / "android/app/src/main/java/com/routemind/app/nav/CopilotStrings.kt")
    sw = parse_swift(app_root / "ios/Runner/MapboxNavigationViewController.swift")
    be = parse_backend(backend_root)

    errors = []
    if not kt:
        errors.append("Android catalog parsed EMPTY — parser or file problem")
    if not sw:
        errors.append("iOS catalog parsed EMPTY — parser or file problem")
    if not be:
        errors.append("backend catalog parsed EMPTY")

    only_kt = set(kt) - set(sw)
    only_sw = set(sw) - set(kt)
    if only_kt:
        errors.append(f"keys on Android but MISSING on iOS: {sorted(only_kt)}")
    if only_sw:
        errors.append(f"keys on iOS but MISSING on Android: {sorted(only_sw)}")

    check_values("android", kt, errors)
    check_values("ios", sw, errors)
    check_values("backend", be, errors)

    print(f"android={len(kt)} keys, ios={len(sw)} keys, backend={len(be)} keys")
    if errors:
        print("\nPARITY FAILURES:")
        for e in errors:
            print("  X " + e)
        return 1
    print("parity OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
