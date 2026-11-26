from gateway.providers.tei import (
    DEFAULT_TEI_MODEL,
    load_tei_config,
)


def test_loads_default_tei_config(monkeypatch):
    for variable in (
        "TEI_BASE_URL",
        "TEI_MODEL",
        "TEI_API_KEY",
        "TEI_TIMEOUT_SECONDS",
        "TEI_DIMENSIONS",
        "TEI_BATCH_SIZE",
    ):
        monkeypatch.delenv(
            variable,
            raising=False,
        )

    config = load_tei_config()

    assert config.embedding.provider == (
        "openai_compatible"
    )
    assert config.embedding.model == (
        DEFAULT_TEI_MODEL
    )
    assert config.embedding.base_url == (
        "http://tei:8080/v1"
    )
    assert config.embedding.dimensions == 384
    assert config.health_url == (
        "http://tei:8080/health"
    )


def test_loads_custom_tei_config(monkeypatch):
    monkeypatch.setenv(
        "TEI_BASE_URL",
        "http://tei-custom:9000/",
    )
    monkeypatch.setenv(
        "TEI_MODEL",
        "custom-model",
    )
    monkeypatch.setenv(
        "TEI_DIMENSIONS",
        "768",
    )

    config = load_tei_config()

    assert config.embedding.model == "custom-model"
    assert config.embedding.base_url == (
        "http://tei-custom:9000/v1"
    )
    assert config.embedding.dimensions == 768