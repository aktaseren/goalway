"""
Shared fixtures for the test suite.

Several gateway modules keep module-level global state (API key
store, usage counters, cache backend, feedback store). Tests that
touch them must reset the globals before and restore them after so
they never leak into other test files.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

import gateway.cache
import gateway.keys
import gateway.rag.feedback as feedback_module


@pytest.fixture
def clean_keys():
    """Reset the API key store + usage counters, restore afterwards."""
    orig_store = gateway.keys._key_store
    orig_usage = gateway.keys._usage

    gateway.keys._key_store = {}
    gateway.keys._usage = {}

    yield gateway.keys

    gateway.keys._key_store = orig_store
    gateway.keys._usage = orig_usage


@pytest.fixture
def clean_cache():
    """Reset the response cache backend + in-memory cache."""
    gateway.cache._backend = None
    gateway.cache.RESPONSE_CACHE.clear()

    yield gateway.cache

    gateway.cache._backend = None
    gateway.cache.RESPONSE_CACHE.clear()


@pytest.fixture
def clean_feedback():
    """Clear the module-global in-memory feedback store."""
    feedback_module._memory_store.clear()

    yield feedback_module

    feedback_module._memory_store.clear()
