import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient

from gateway.main import app

client = TestClient(app)


def test_ui_served():
    response = client.get("/ui")

    assert response.status_code == 200
    assert "Football News AI" in response.text
    assert "text/html" in response.headers["content-type"]


def test_root_advertises_ui():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["ui"] == "/ui"
