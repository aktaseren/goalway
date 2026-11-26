"""Tests for gateway/keys.py (API key store, roles, quotas)."""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

import gateway.keys


KEY_FILE_CONTENT = {
    "admin-key-12345678": {
        "name": "admin",
        "role": "admin",
        "quota": None,
    },
    "limited-key-12345678": {
        "name": "limited-user",
        "role": "user",
        "quota": 2,
    },
    "abc": {
        # no name/role -> defaults
        "quota": 5,
    },
}


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps(KEY_FILE_CONTENT))
    return str(path)


def test_load_api_keys_from_file(clean_keys, key_file):
    count = gateway.keys.load_api_keys(key_file)

    assert count == 3


def test_lookup_key_roles_and_quotas(clean_keys, key_file):
    gateway.keys.load_api_keys(key_file)

    admin = gateway.keys.lookup_key("admin-key-12345678")
    assert admin == {"name": "admin", "role": "admin", "quota": None}

    limited = gateway.keys.lookup_key("limited-key-12345678")
    assert limited == {
        "name": "limited-user",
        "role": "user",
        "quota": 2,
    }

    # Missing fields fall back to defaults.
    plain = gateway.keys.lookup_key("abc")
    assert plain["role"] == "user"
    assert plain["name"] == "abc"
    assert plain["quota"] == 5


def test_lookup_unknown_key(clean_keys, key_file):
    gateway.keys.load_api_keys(key_file)

    assert gateway.keys.lookup_key("no-such-key") is None


def test_load_api_keys_missing_file_falls_back_to_legacy(clean_keys):
    count = gateway.keys.load_api_keys(
        "/nonexistent/path/api_keys.json"
    )

    # Legacy single admin key.
    assert count == 1
    entry = gateway.keys.lookup_key(gateway.keys.API_KEY)
    assert entry is not None
    assert entry["role"] == "admin"


def test_load_api_keys_invalid_json_falls_back(clean_keys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")

    count = gateway.keys.load_api_keys(str(bad))

    assert count == 1
    assert gateway.keys.lookup_key(gateway.keys.API_KEY) is not None


def test_lookup_key_triggers_load(clean_keys):
    # Empty store -> lookup lazily loads the configured key file.
    gateway.keys._key_store = {}

    entry = gateway.keys.lookup_key("redhat-demo-key")

    assert entry is not None
    assert entry["role"] == "admin"


def test_quota_enforcement(clean_keys):
    gateway.keys._key_store = {
        "limited-key": {
            "name": "limited-user",
            "role": "user",
            "quota": 2,
        }
    }
    gateway.keys._usage = {}

    gateway.keys.check_quota("limited-key")
    gateway.keys.check_quota("limited-key")

    with pytest.raises(gateway.keys.QuotaExceededError) as excinfo:
        gateway.keys.check_quota("limited-key")

    assert excinfo.value.limit == 2
    assert excinfo.value.key_name == "limited-user"
    assert excinfo.value.reset_in_seconds >= 0


def test_quota_unlimited_key_never_raises(clean_keys):
    gateway.keys._key_store = {
        "admin-key": {"name": "admin", "role": "admin", "quota": None}
    }
    gateway.keys._usage = {}

    for _ in range(20):
        gateway.keys.check_quota("admin-key")


def test_quota_unknown_key_is_noop(clean_keys):
    gateway.keys._key_store = {}
    gateway.keys.check_quota("unknown-key")  # must not raise


def test_quota_window_reset(monkeypatch, clean_keys):
    gateway.keys._key_store = {
        "limited-key": {
            "name": "limited-user",
            "role": "user",
            "quota": 1,
        }
    }
    gateway.keys._usage = {}

    window = 1000
    monkeypatch.setattr(gateway.keys, "QUOTA_WINDOW_SECONDS", window)

    current = {"now": 10_000.0}
    monkeypatch.setattr(
        gateway.keys.time, "time", lambda: current["now"]
    )

    # First call in window: passes, count = 1.
    gateway.keys.check_quota("limited-key")

    # Same window: quota exhausted.
    with pytest.raises(gateway.keys.QuotaExceededError):
        gateway.keys.check_quota("limited-key")

    # After the window rolls over the counter resets.
    current["now"] = 10_000.0 + window + 1
    gateway.keys.check_quota("limited-key")


def test_list_keys_hides_secret(clean_keys, key_file):
    gateway.keys.load_api_keys(key_file)

    keys = gateway.keys.list_keys()

    assert len(keys) == 3
    serialized = json.dumps(keys)
    assert "admin-key-12345678" not in serialized
    assert "limited-key-12345678" not in serialized

    by_name = {entry["name"]: entry for entry in keys}
    assert by_name["admin"]["role"] == "admin"
    assert by_name["admin"]["quota"] is None
    assert "key_preview" in by_name["admin"]

    # Short keys are fully masked.
    assert by_name["abc"]["key_preview"] == "***"


def test_quota_usage_shape(clean_keys):
    gateway.keys._key_store = {
        "limited-key": {
            "name": "limited-user",
            "role": "user",
            "quota": 7,
        }
    }
    gateway.keys._usage = {
        "limited-key": {"window_start": 1.0, "count": 3}
    }

    usage = gateway.keys.quota_usage()

    assert usage["limited-user"]["role"] == "user"
    assert usage["limited-user"]["limit"] == 7
    assert usage["limited-user"]["used"] == 3
    assert usage["limited-user"]["window_seconds"] == (
        gateway.keys.QUOTA_WINDOW_SECONDS
    )
