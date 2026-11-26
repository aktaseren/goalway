import time
import requests

from gateway.metrics import RETRIES

CIRCUITS = {}

FAILURE_THRESHOLD = 3
RECOVERY_TIMEOUT = 30

def post_with_retries(
    url:str,
    payload: dict,
    headers: dict,
    timeout: int = 600,
    max_retries: int = 2,
    backoff_seconds: float = 1.0
):

    last_error = None

    for attempt in range(max_retries +1):
        try:
            response = requests.post(
                url,
                json = payload,
                headers = headers,
                timeout = timeout,
            )
            response.raise_for_status()
            return response, attempt
        except requests.exceptions.RequestException as e:
            last_error = e

            if attempt < max_retries:
                RETRIES.inc()
                time.sleep(backoff_seconds * (attempt + 1))
    raise last_error

def is_circuit_open(backend_name: str):
    circuit = CIRCUITS.get(backend_name)

    if not circuit:
        return False
    
    if not circuit["open"]:
        return False
    
    age = time.time() - circuit["opened at"]

    if age >= RECOVERY_TIMEOUT:
        circuit["open"] = False
        circuit["failures"] = 0
        return False
    
    return True

def record_success(backend_name: str):
    CIRCUITS[backend_name] = {
        "failures": 0,
        "open": False,
        "opened_at": 0,
    }

def record_failure(backend_name: str):
    circuit = CIRCUITS.setdefault(
        backend_name,
        {
            "failures":0,
            "open": False,
            "opened_at": 0,
        },
    )

    circuit["failures"] += 1

    if circuit["failures"] >= FAILURE_THRESHOLD:
        circuit["open"] = True
        circuit["opened_at"] = time.time()