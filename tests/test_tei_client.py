from unittest.mock import Mock, patch

import requests

from gateway.providers.tei import (
    TEIClient,
    load_tei_config,
)


@patch(
    "gateway.providers.tei.client.requests.get"
)
def test_tei_health_returns_true(mock_get):
    response = Mock()
    response.status_code = 200
    mock_get.return_value = response

    client = TEIClient(
        load_tei_config()
    )

    assert client.health() is True


@patch(
    "gateway.providers.tei.client.requests.get"
)
def test_tei_health_returns_false_on_error(
    mock_get,
):
    mock_get.side_effect = (
        requests.ConnectionError()
    )

    client = TEIClient(
        load_tei_config()
    )

    assert client.health() is False


@patch(
    "gateway.providers.tei.client.requests.get"
)
def test_tei_info(mock_get):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "model_id": "granite-embedding",
        "model_dtype": "float32",
    }
    mock_get.return_value = response

    client = TEIClient(
        load_tei_config()
    )

    info = client.info()

    assert info["model_id"] == (
        "granite-embedding"
    )