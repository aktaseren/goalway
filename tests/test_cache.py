import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gateway.cache import make_cache_key

def test_cache_key_is_stable():
    a = make_cache_key("echo", [{"role": "user", "content": "Hi"}])
    b = make_cache_key("echo", [{"role": "user", "content": "Hi"}])
    assert a == b