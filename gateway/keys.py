"""
API key store with roles and per-key quotas .

Keys live in a JSON file (mount it as a ConfigMap in OpenShift and
reload with POST /admin/reload). Each entry defines a display name,
an RBAC role, and an optional request quota per window.

Example config/api_keys.json:

{
  "redhat-demo-key": {
    "name": "admin",
    "role": "admin",
    "quota": null
  },
  "team-user-key": {
    "name": "platform-team",
    "role": "user",
    "quota": 100
  }
}
"""

import json
import logging
import threading
import time

from gateway.config import (
    API_KEYS_FILE,
    API_KEY,
    DEFAULT_QUOTA_REQUESTS,
    QUOTA_WINDOW_SECONDS,
)

logger = logging.getLogger("goalway.keys")

# Legacy fallback when no key store file exists: the original single
# admin key from V1-keeps working.
FALLBACK_KEYS = {
    API_KEY: {
        "name": "admin",
        "role": "admin",
        "quota": None,
    },
}

_lock = threading.Lock()
_key_store: dict = {}
_usage: dict = {}  # api_key -> {window_start, count}


class QuotaExceededError(Exception):
    def __init__(self, key_name: str, limit: int, reset_in_seconds: int):
        self.key_name = key_name
        self.limit = limit
        self.reset_in_seconds = reset_in_seconds
        super().__init__(f"Quota exceeded for {key_name}")


def load_api_keys(path: str | None = None) -> int:
    """(Re)load the key store from disk. Returns number of keys."""
    global _key_store

    # Evaluated at call time (not import time) so tests and admin
    # reloads can repoint API_KEYS_FILE via the environment.
    if path is None:
        path = API_KEYS_FILE

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.info(
            "API key store %s not found, using legacy single key",
            path,
        )
        _key_store = dict(FALLBACK_KEYS)
        return len(_key_store)
    except json.JSONDecodeError as exc:
        logger.error("Invalid API key store %s: %s", path, exc)
        _key_store = dict(FALLBACK_KEYS)
        return len(_key_store)

    _key_store = {}
    for key, entry in data.items():
        _key_store[key] = {
            "name": entry.get("name", key),
            "role": entry.get("role", "user"),
            "quota": entry.get("quota"),
        }

    logger.info("loaded %d API keys from %s", len(_key_store), path)
    return len(_key_store)


def lookup_key(api_key: str) -> dict | None:
    """Return {name, role, quota} for a key, or None if unknown."""
    if not _key_store:
        load_api_keys()

    return _key_store.get(api_key)


def list_keys() -> list[dict]:
    """Admin view: key metadata without the secrets themselves."""
    if not _key_store:
        load_api_keys()

    return [
        {
            "name": entry["name"],
            "role": entry["role"],
            "quota": entry["quota"],
            "key_preview": f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***",
        }
        for key, entry in _key_store.items()
    ]


def check_quota(api_key: str) -> None:
    """
    Count a request against the key's quota.

    Raises QuotaExceededError when the key has exhausted its quota
    for the current window.
    """
    entry = lookup_key(api_key)

    if entry is None:
        return

    limit = entry.get("quota")

    if limit is None:
        limit = DEFAULT_QUOTA_REQUESTS

    now = time.time()

    with _lock:
        usage = _usage.get(api_key)

        if usage is None or now - usage["window_start"] > QUOTA_WINDOW_SECONDS:
            usage = {
                "window_start": now,
                "count": 0,
            }
            _usage[api_key] = usage

        if usage["count"] >= limit:
            raise QuotaExceededError(
                key_name=entry["name"],
                limit=limit,
                reset_in_seconds=int(
                    QUOTA_WINDOW_SECONDS - (now - usage["window_start"])
                ),
            )

        usage["count"] += 1


def quota_usage() -> dict:
    """Admin view: current usage per key."""
    if not _key_store:
        load_api_keys()

    result = {}

    for key, entry in _key_store.items():
        usage = _usage.get(key)
        limit = entry.get("quota") or DEFAULT_QUOTA_REQUESTS

        result[entry["name"]] = {
            "role": entry["role"],
            "limit": limit,
            "used": usage["count"] if usage else 0,
            "window_seconds": QUOTA_WINDOW_SECONDS,
        }

    return result
