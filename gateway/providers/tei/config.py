import os
from dataclasses import dataclass

from gateway.rag.embeddings.config import EmbeddingConfig


DEFAULT_TEI_MODEL = (
    "ibm-granite/granite-embedding-small-english-r2"
)


@dataclass(frozen=True)
class TEIConfig:
    embedding: EmbeddingConfig
    health_url: str
    info_url: str


def load_tei_config() -> TEIConfig:
    base_url = os.getenv(
        "TEI_BASE_URL",
        "http://tei:8080",
    ).rstrip("/")

    return TEIConfig(
        embedding=EmbeddingConfig(
            provider="openai_compatible",
            model=os.getenv(
                "TEI_MODEL",
                DEFAULT_TEI_MODEL,
            ),
            base_url=f"{base_url}/v1",
            api_key=os.getenv("TEI_API_KEY"),
            timeout_seconds=int(
                os.getenv(
                    "TEI_TIMEOUT_SECONDS",
                    "120",
                )
            ),
            dimensions=int(
                os.getenv(
                    "TEI_DIMENSIONS",
                    "384",
                )
            ),
            batch_size=int(
                os.getenv(
                    "TEI_BATCH_SIZE",
                    "32",
                )
            ),
            extra={
                "runtime": "tei",
            },
        ),
        health_url=f"{base_url}/health",
        info_url=f"{base_url}/info",
    )