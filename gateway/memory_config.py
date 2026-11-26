import os

MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "memory").lower()


def _parse_port(value: str, default: str) -> int:
    """
    Parse a port from an env var that may also hold a Kubernetes
    service-link value like "tcp://10.217.5.226:6379" (injected for
    every Service in the namespace, e.g. one named "redis").
    """
    value = value.strip()

    if value.startswith("tcp://") or value.startswith("://"):
        value = value.split("://", 1)[1]

    if ":" in value:
        value = value.rsplit(":", 1)[1]

    try:
        return int(value)
    except ValueError:
        return int(default)


def _redis_from_url(url: str) -> tuple[str, int]:
    """Extract host and port from redis://host:port[/db]."""
    rest = url.split("://", 1)[1] if "://" in url else url
    rest = rest.split("/", 1)[0]

    if "@" in rest:  # strip credentials if present
        rest = rest.rsplit("@", 1)[1]

    host, _, port = rest.rpartition(":")

    return (host or "redis", _parse_port(port or "6379", "6379"))


# Prefer one REDIS_URL (also used by the response cache) over the
# separate host/port vars.
REDIS_URL = os.getenv("REDIS_URL", "")

if REDIS_URL:
    REDIS_HOST, REDIS_PORT = _redis_from_url(REDIS_URL)
else:
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = _parse_port(os.getenv("REDIS_PORT", "6379"), "6379")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = _parse_port(os.getenv("POSTGRES_PORT", "5432"), "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "aigateway")
POSTGRES_USER = os.getenv("POSTGRES_USER", "aiuser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "aipassword")
