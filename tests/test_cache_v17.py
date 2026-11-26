"""Tests for gateway/cache.py backends (in-memory + Redis)."""

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

import gateway.cache
from gateway.cache import (
    InMemoryCacheBackend,
    RedisCacheBackend,
    get_cache_backend,
    get_cached_response,
    make_cache_key,
    store_cached_response,
)


class FakeRedisClient:
    """Dict-backed stand-in for redis.Redis (str values only)."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.deleted: list[str] = []

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None):
        self.store[key] = value

    def delete(self, key: str):
        self.deleted.append(key)
        self.store.pop(key, None)


def test_make_cache_key_deterministic_and_distinct():
    messages = [{"role": "user", "content": "hello"}]

    key_1 = make_cache_key("echo", messages)
    key_2 = make_cache_key("echo", messages)
    key_3 = make_cache_key("granite", messages)

    assert key_1 == key_2
    assert key_1 != key_3
    assert len(key_1) == 64  # sha256 hex digest


def test_cache_backend_failure_degrades_to_miss(clean_cache):
    """A broken cache backend must never fail requests (CRC incident)."""
    import gateway.cache as cache_mod

    class BrokenBackend:
        def get(self, key, ttl):
            raise RuntimeError("MISCONF Redis is unable to persist")

        def set(self, key, data, ttl):
            raise RuntimeError("MISCONF Redis is unable to persist")

        def name(self):
            return "broken"

    backend = BrokenBackend()
    cache_mod._backend = backend

    # get degrades to None (miss) instead of raising
    assert cache_mod.get_cached_response("k", 60) is None

    # store is best-effort and must not raise
    cache_mod.store_cached_response("k", {"gateway": {}, "choices": []})


def test_in_memory_backend_set_get(clean_cache):
    backend = InMemoryCacheBackend()

    backend.set("k1", {"answer": 42}, ttl_seconds=100)

    assert backend.get("k1", 100) == {"answer": 42}
    assert backend.name() == "memory"


def test_in_memory_backend_ttl_expiry(clean_cache):
    backend = InMemoryCacheBackend()

    backend.set("k1", {"answer": 42}, ttl_seconds=100)

    # Age the entry beyond the TTL.
    gateway.cache.RESPONSE_CACHE["k1"]["created_at"] = time.time() - 101

    assert backend.get("k1", 100) is None
    assert "k1" not in gateway.cache.RESPONSE_CACHE


def test_in_memory_backend_miss(clean_cache):
    assert InMemoryCacheBackend().get("missing", 100) is None


def test_redis_backend_roundtrip():
    backend = RedisCacheBackend.__new__(RedisCacheBackend)
    backend.url = "redis://fake"
    backend.client = FakeRedisClient()

    backend.set("k1", {"answer": 42}, ttl_seconds=100)

    stored = backend.client.store["cache:k1"]
    payload = json.loads(stored)
    assert payload["data"] == {"answer": 42}
    assert "created_at" in payload

    assert backend.get("k1", 100) == {"answer": 42}
    assert backend.name() == "redis"


def test_redis_backend_ttl_expiry():
    backend = RedisCacheBackend.__new__(RedisCacheBackend)
    backend.url = "redis://fake"
    backend.client = FakeRedisClient()

    backend.set("k1", {"answer": 42}, ttl_seconds=100)
    backend.client.store["cache:k1"] = json.dumps(
        {"created_at": time.time() - 101, "data": {"answer": 42}}
    )

    assert backend.get("k1", 100) is None
    assert "cache:k1" not in backend.client.store
    assert "cache:k1" in backend.client.deleted


def test_redis_backend_invalid_json_deleted():
    backend = RedisCacheBackend.__new__(RedisCacheBackend)
    backend.url = "redis://fake"
    backend.client = FakeRedisClient()

    backend.client.store["cache:k1"] = "not json {{{"

    assert backend.get("k1", 100) is None
    assert "cache:k1" not in backend.client.store


def test_redis_backend_miss():
    backend = RedisCacheBackend.__new__(RedisCacheBackend)
    backend.url = "redis://fake"
    backend.client = FakeRedisClient()

    assert backend.get("missing", 100) is None


def test_factory_picks_memory_when_no_redis_url(clean_cache, monkeypatch):
    monkeypatch.setattr(gateway.cache, "REDIS_URL", "")

    backend = get_cache_backend()

    assert isinstance(backend, InMemoryCacheBackend)
    assert backend.name() == "memory"


def test_factory_falls_back_to_memory_on_redis_failure(
    clean_cache, monkeypatch
):
    monkeypatch.setattr(gateway.cache, "REDIS_URL", "redis://unreachable:1")

    class BrokenRedis:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("redis down")

    monkeypatch.setattr(gateway.cache, "RedisCacheBackend", BrokenRedis)

    backend = get_cache_backend()

    assert isinstance(backend, InMemoryCacheBackend)
    assert backend.name() == "memory"


def test_factory_caches_backend(clean_cache, monkeypatch):
    monkeypatch.setattr(gateway.cache, "REDIS_URL", "")

    first = get_cache_backend()
    second = get_cache_backend()

    assert first is second


def test_get_cached_response_marks_hit(clean_cache, monkeypatch):
    monkeypatch.setattr(gateway.cache, "REDIS_URL", "")

    response = {
        "choices": [{"message": {"content": "cached"}}],
        "gateway": {"model": "echo"},
    }

    key = make_cache_key("echo", [{"role": "user", "content": "hi"}])
    store_cached_response(key, response)

    cached = get_cached_response(key, ttl_seconds=100)

    assert cached is not None
    assert cached["gateway"]["cache"] == "hit"
    assert cached["gateway"]["cache_backend"] == "memory"
    assert cached["choices"][0]["message"]["content"] == "cached"


def test_get_cached_response_miss(clean_cache, monkeypatch):
    monkeypatch.setattr(gateway.cache, "REDIS_URL", "")

    assert get_cached_response("no-such-key", 100) is None
