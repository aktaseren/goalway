import time
from collections import defaultdict, deque

from fastapi import HTTPException

REQUEST_HISTORY = defaultdict(deque)

def check_rate_limit(client_key: str, limit: int, window_seconds: int):
    now = time.time()
    history = REQUEST_HISTORY[client_key]

    while history and now - history[0] > window_seconds:
        history.popleft()

    if len(history) >= limit:
        retry_after = int(window_seconds - (now - history[0]))
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after_seconds": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )

    history.append(now)