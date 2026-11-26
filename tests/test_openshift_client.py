from unittest.mock import Mock, patch

import pytest

from gateway.providers.openshift import (
    OpenShiftAIClient,
)


@patch(
    "gateway.providers.openshift.client.requests.get"
)
def test_lists_openshift_models(mock_get):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "granite-3",
                "object": "model",
            },
            {
                "id": "embedding-model",
                "object": "model",
            },
        ],
    }
    mock_get.return_value = response

    client = OpenShiftAIClient(
        endpoint="http://model-service:8000/v1",
        api_key="secret",
    )

    models = client.list_models()

    assert [model.name for model in models] == [
        "granite-3",
        "embedding-model",
    ]

    mock_get.assert_called_once_with(
        "http://model-service:8000/v1/models",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer secret",
        },
        timeout=30,
    )


@patch(
    "gateway.providers.openshift.client.requests.get"
)
def test_creates_generation_config(mock_get):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [
            {
                "id": "granite-3",
                "owned_by": "red-hat",
            }
        ],
    }
    mock_get.return_value = response

    client = OpenShiftAIClient(
        endpoint="http://model-service:8000/v1",
        api_key="secret",
        timeout_seconds=90,
    )

    config = client.create_generation_config(
        "granite-3"
    )

    assert config.provider == "openshift_ai"
    assert config.model == "granite-3"
    assert config.base_url == (
        "http://model-service:8000/v1"
    )
    assert config.api_key == "secret"
    assert config.timeout_seconds == 90
    assert config.extra["discovered"] is True
    assert config.extra["owned_by"] == "red-hat"


@patch(
    "gateway.providers.openshift.client.requests.get"
)
def test_missing_model_is_rejected(mock_get):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [
            {
                "id": "granite-3",
            }
        ],
    }
    mock_get.return_value = response

    client = OpenShiftAIClient(
        endpoint="http://model-service:8000/v1",
    )

    with pytest.raises(
        ValueError,
        match="model not found",
    ):
        client.get_model("missing-model")


@patch(
    "gateway.providers.openshift.client.requests.get"
)
def test_health_returns_true_for_healthy_endpoint(
    mock_get,
):
    response = Mock()
    response.status_code = 200
    mock_get.return_value = response

    client = OpenShiftAIClient(
        endpoint="http://model-service:8000/v1",
    )

    assert client.health() is True


@patch(
    "gateway.providers.openshift.client.requests.get"
)
def test_health_returns_false_on_connection_error(
    mock_get,
):
    import requests

    mock_get.side_effect = requests.ConnectionError()

    client = OpenShiftAIClient(
        endpoint="http://model-service:8000/v1",
    )

    assert client.health() is False