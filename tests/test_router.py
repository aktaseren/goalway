import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gateway.api import choose_model

def test_auto_routes_hi_to_echo():
    assert choose_model("Hi") == "echo"

def test_auto_routes_openshift_to_granite():
    assert choose_model("Explain OpenShift routes") == "granite"