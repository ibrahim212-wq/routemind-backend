"""
services/supabase_client.py

Direct HTTP wrapper للـ Supabase REST API.
بيتجنب الـ SDK validation issues مع الـ sb_ key format.
"""

import os
import logging
from urllib.parse import quote
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger("routemind.supabase")


class SupabaseClient:
    def __init__(self, url: str, key: str):
        self.base_url = f"{url.rstrip('/')}/rest/v1"
        self.headers = {
            "apikey":        key,
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
            "Prefer":        "return=representation",
        }

    def table(self, name: str) -> "TableQuery":
        return TableQuery(self.base_url, self.headers, name)


class TableQuery:
    def __init__(self, base_url: str, headers: dict, table: str):
        self._base_url    = base_url
        self._headers     = headers
        self._table       = table
        self._filters: List[tuple] = []     # (col, operator_value) tuples
        self._select_cols = "*"
        self._limit_val: Optional[int] = None
        self._order_col: Optional[str] = None
        self._order_desc  = False
        self._single      = False
        self._update_data: Optional[Dict] = None

    # ── Builder methods ────────────────────────────────────────

    def select(self, cols: str = "*") -> "TableQuery":
        self._select_cols = cols
        return self

    def eq(self, col: str, val: Any) -> "TableQuery":
        self._filters.append((col, f"eq.{val}"))
        return self

    def gte(self, col: str, val: Any) -> "TableQuery":
        self._filters.append((col, f"gte.{val}"))
        return self

    def lte(self, col: str, val: Any) -> "TableQuery":
        self._filters.append((col, f"lte.{val}"))
        return self

    def limit(self, n: int) -> "TableQuery":
        self._limit_val = n
        return self

    def order(self, col: str, desc: bool = False) -> "TableQuery":
        self._order_col  = col
        self._order_desc = desc
        return self

    def single(self) -> "TableQuery":
        self._single = True
        return self

    def update(self, data: Dict) -> "TableQuery":
        self._update_data = data
        return self

    # ── URL builder ────────────────────────────────────────────

    def _build_url(self) -> str:
        url    = f"{self._base_url}/{self._table}"
        params = [f"select={quote(self._select_cols, safe='*,()')}"]

        for col, op_val in self._filters:
            # URL-encode the value part. safe='.' keeps the operator dot
            # (eq./gte./lte.) readable but encodes + : etc in timestamps.
            encoded = quote(op_val, safe=".")
            params.append(f"{col}={encoded}")

        if self._order_col:
            direction = "desc" if self._order_desc else "asc"
            params.append(f"order={self._order_col}.{direction}")
        if self._limit_val:
            params.append(f"limit={self._limit_val}")

        return url + "?" + "&".join(params)

    # ── Terminal methods ───────────────────────────────────────

    def execute(self) -> "Result":
        url     = self._build_url()
        headers = dict(self._headers)
        if self._single:
            headers["Accept"] = "application/vnd.pgrst.object+json"
        try:
            resp = httpx.get(url, headers=headers, timeout=10.0)
            resp.raise_for_status()
            return Result(resp.json())
        except Exception as e:
            logger.error(f"Supabase GET {self._table}: {e}")
            return Result(None, error=str(e))

    def insert(self, data: Any) -> "Result":
        url = f"{self._base_url}/{self._table}"
        try:
            resp = httpx.post(url, headers=self._headers,
                              json=data, timeout=10.0)
            resp.raise_for_status()
            return Result(resp.json() if resp.text else [])
        except Exception as e:
            logger.error(f"Supabase INSERT {self._table}: {e}")
            return Result(None, error=str(e))

    def execute_update(self) -> "Result":
        url     = self._build_url()
        headers = dict(self._headers)
        headers["Prefer"] = "return=minimal"
        try:
            resp = httpx.patch(url, headers=headers,
                               json=self._update_data, timeout=10.0)
            resp.raise_for_status()
            return Result([])
        except Exception as e:
            logger.error(f"Supabase UPDATE {self._table}: {e}")
            return Result(None, error=str(e))

    # backwards-compat alias
    def _execute_update(self) -> "Result":
        return self.execute_update()


class Result:
    def __init__(self, data: Any, error: Optional[str] = None):
        self.data  = data
        self.error = error


def get_supabase() -> Optional[SupabaseClient]:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        logger.warning("SUPABASE_URL أو SUPABASE_SERVICE_KEY مش موجود")
        return None
    return SupabaseClient(url, key)