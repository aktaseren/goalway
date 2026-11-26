import json
import time
from fastapi.responses import StreamingResponse
import requests

def fake_stream_response(text: str, delay: float = 0.05):
    words = text.split()

    for word in words:
        chunk = {
            "choices": [
                {
                    "delta": {
                        "content": word + " "
                    }
                }
            ]
        }

        yield f"data: {json.dumps(chunk)}\n\n"
        time.sleep(delay)

    yield "data: [DONE]\n\n"

def stream_response(text:str):
    return StreamingResponse(
        fake_stream_response(text),
        media_type="text/event-stream",
    )

def backend_stream_response(
    backend_url: str,
    payload: dict,
    delay: float = 0.03,
):
    response = requests.post(
        backend_url,
        json = payload,
        timeout = 600,
    )

    response.raise_for_status()

    data = response.json()

    text = data["choices"][0]["message"]["content"]

    yield from fake_stream_response(text, delay)

def stream_backend_response(
    backend_url: str,
    payload: dict,
):
    return StreamingResponse(
        backend_stream_response(
            backend_url,
            payload,
        ),
        media_type = "text/event-stream",
    )
