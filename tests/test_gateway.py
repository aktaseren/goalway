import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gateway.api import choose_model

def test_choose_model_returns_valid_backend():
    model = choose_model("Hello")
    assert model in ["echo", "tinyllama", "granite"]