import os

from gateway.providers.generation import (
    GenerationProvider,
    create_generation_provider,
)
from gateway.providers.openshift.client import (
    OpenShiftAIClient,
)


def create_openshift_generation_provider(
    model_name: str | None = None,
) -> GenerationProvider:
    endpoint = os.getenv(
        "OPENSHIFT_AI_BASE_URL",
        "http://granite-service:8080/v1",
    )

    api_key = os.getenv(
        "OPENSHIFT_AI_API_KEY"
    )

    timeout_seconds = int(
        os.getenv(
            "OPENSHIFT_AI_TIMEOUT_SECONDS",
            "600",
        )
    )

    resolved_model = (
        model_name
        or os.getenv(
            "OPENSHIFT_AI_MODEL",
            "granite",
        )
    )

    client = OpenShiftAIClient(
        endpoint=endpoint,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

    config = client.create_generation_config(
        resolved_model
    )

    return create_generation_provider(config)