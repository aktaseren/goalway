import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gateway.health import health_response

def test_health_response_has_status():
    result = health_response()
    assert result["status"] == "healthy"
