import hashlib
import json
import logging
import time

from gateway.config import CACHE_TTL_SECONDS, REDIS_URL
from gateway.metrics import CACHE_ERRORS

logger = logging.getLogger("goalway.cache")

RESPONSE_CACHE = {}


class CacheBackend:
    """Interface for response cache backends."""

    def get(self, cache_key: str, ttl_seconds: int):
        raise NotImplementedError

    def set(self, cache_key: str, data: dict, ttl_seconds: int):
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


class InMemoryCacheBackend(CacheBackend):
    """Original V7 TTL cache, process-local."""

    def get(self, cache_key: str, ttl_seconds: int):
        if cache_key not in RESPONSE_CACHE:
            return None

        cached_item = RESPONSE_CACHE[cache_key]
        age = time.time() - cached_item["created_at"]

        if age > ttl_seconds:
            del RESPONSE_CACHE[cache_key]
            return None

        return cached_item["data"]

    def set(self, cache_key: str, data: dict, ttl_seconds: int):
        RESPONSE_CACHE[cache_key] = {
            "created_at": time.time(),
            "data": data,
        }

    def name(self) -> str:
        return "memory"


class RedisCacheBackend(CacheBackend):
    """
    Shared, cross-replica response cache backed by Redis .

    Values are stored as JSON with a created_at timestamp so the
    TTL logic stays identical to the in-memory backend.
    """

    def __init__(self, redis_url: str):
        import redis  # lazy: already a core dependency

        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )

    def get(self, cache_key: str, ttl_seconds: int):
        raw = self.client.get(f"cache:{cache_key}")

        if raw is None:
            return None

        try:
            cached_item = json.loads(raw)
        except json.JSONDecodeError:
            self.client.delete(f"cache:{cache_key}")
            return None

        age = time.time() - cached_item.get("created_at", 0)

        if age > ttl_seconds:
            self.client.delete(f"cache:{cache_key}")
            return None

        return cached_item.get("data")

    def set(self, cache_key: str, data: dict, ttl_seconds: int):
        payload = json.dumps(
            {
                "created_at": time.time(),
                "data": data,
            }
        )
        # Keep entries a bit longer than the logical TTL so the TTL
        # check above stays the single source of truth.
        self.client.set(
            f"cache:{cache_key}",
            payload,
            ex=ttl_seconds * 2,
        )

    def name(self) -> str:
        return "redis"


_backend: CacheBackend | None = None


def get_cache_backend() -> CacheBackend:
    """Return the configured cache backend (Redis when available)."""
    global _backend

    if _backend is not None:
        return _backend

    if REDIS_URL:
        try:
            _backend = RedisCacheBackend(REDIS_URL)
            logger.info("response cache backend: redis")
        except Exception as exc:
            logger.warning(
                "Redis cache unavailable (%s), falling back to memory",
                exc,
            )
            _backend = InMemoryCacheBackend()
    else:
        _backend = InMemoryCacheBackend()

    return _backend


def make_cache_key(model: str, messages: list) -> str:
    """
    Create a unique cache key based on the model and request data.
    """
    raw = json.dumps(
        {
            "model": model,
            "messages": messages,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_response(cache_key: str, ttl_seconds: int):
    backend = get_cache_backend()

    try:
        cached_data = backend.get(cache_key, ttl_seconds)
    except Exception as exc:
        # A cache outage must never fail the request: degrade to a
        # cache miss and keep serving.
        logger.warning("cache get failed (%s), treating as miss", exc)
        CACHE_ERRORS.inc()

        return None

    if cached_data is not None:
        cached_data["gateway"]["cache"] = "hit"
        cached_data["gateway"]["cache_backend"] = backend.name()

    return cached_data


def store_cached_response(cache_key: str, data: dict):
    try:
        get_cache_backend().set(cache_key, data, CACHE_TTL_SECONDS)
    except Exception as exc:
        # Same contract on writes: best-effort caching.
        logger.warning("cache set failed (%s), skipping cache store", exc)
        CACHE_ERRORS.inc()
