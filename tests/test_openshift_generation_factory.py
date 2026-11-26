from unittest.mock import patch

from gateway.providers.generation import (
    OpenAICompatibleGenerationProvider,
)
from gateway.providers.openshift import (
    create_openshift_generation_provider,
)
from gateway.providers.openshift.models import (
    OpenShiftModel,
)


@patch(
    "gateway.providers.openshift.factory."
    "OpenShiftAIClient.get_model"
)
def test_creates_openshift_generation_provider(
    mock_get_model,
    monkeypatch,
):
    mock_get_model.return_value = OpenShiftModel(
        name="granite-3",
        url="http://granite-service:8000/v1",
        ready=True,
    )

    monkeypatch.setenv(
        "OPENSHIFT_AI_BASE_URL",
        "http://granite-service:8000/v1",
    )
    monkeypatch.setenv(
        "OPENSHIFT_AI_MODEL",
        "granite-3",
    )
    monkeypatch.setenv(
        "OPENSHIFT_AI_API_KEY",
        "secret",
    )

    provider = (
        create_openshift_generation_provider()
    )

    assert isinstance(
        provider,
        OpenAICompatibleGenerationProvider,
    )
    assert provider.config.provider == "openshift_ai"
    assert provider.config.model == "granite-3"