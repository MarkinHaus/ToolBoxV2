"""
Low-level DB operations via TBEF.DB.
All auth state is stored in TBEF.DB — no global dicts, no BlobFile.
"""

import json
from typing import Optional

from toolboxv2 import App, Result, TBEF


def _parse_db_result(raw) -> Optional[dict]:
    """Safely parse a DB result into a dict."""
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return raw if isinstance(raw, dict) else None


async def _db_set(app: App, key: str, data) -> Result:
    """Store data in TBEF.DB."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return await app.a_run_any(TBEF.DB.SET, query=key, data=payload, get_results=True)


async def _db_get(app: App, key: str) -> Optional[dict]:
    """Load data from TBEF.DB, returns parsed dict or None."""
    result = await app.a_run_any(TBEF.DB.GET, query=key, get_results=True)
    if result.is_error():
        return None
    return _parse_db_result(result.get())


async def _db_get_raw(app: App, key: str) -> Optional[str]:
    """Load raw string from TBEF.DB."""
    result = await app.a_run_any(TBEF.DB.GET, query=key, get_results=True)
    if result.is_error():
        return None
    raw = result.get()
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, bytes):
        raw = raw.decode()
    return raw


async def _db_delete(app: App, key: str) -> Result:
    """Delete data from TBEF.DB."""
    return await app.a_run_any(TBEF.DB.DELETE, query=key, get_results=True)


async def _db_exists(app: App, key: str) -> bool:
    """Check if key exists in TBEF.DB."""
    result = await app.a_run_any(TBEF.DB.IF_EXIST, query=key, get_results=True)
    if result.is_error():
        return False
    count = result.get()
    return isinstance(count, int) and count > 0
