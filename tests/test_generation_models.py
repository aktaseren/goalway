import pytest

from gateway.providers.generation import (
    GenerationMessage,
    GenerationRequest,
)


def test_generation_request_accepts_valid_messages():
    request = GenerationRequest(
        messages=[
            GenerationMessage(
                role="user",
                content="Hello",
            )
        ],
        temperature=0.5,
        max_tokens=100,
    )

    assert request.messages[0].content == "Hello"


def test_generation_request_rejects_invalid_temperature():
    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        GenerationRequest(
            messages=[
                GenerationMessage(
                    role="user",
                    content="Hello",
                )
            ],
            temperature=3.0,
        )


def test_generation_request_rejects_empty_messages():
    with pytest.raises(
        ValueError,
        match="at least one message",
    ):
        GenerationRequest(messages=[])