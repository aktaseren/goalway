from unittest.mock import Mock, patch

from gateway.providers.common import ProviderConfig
from gateway.providers.generation import (
    GenerationMessage,
    GenerationRequest,
    OpenAICompatibleGenerationProvider,
)


@patch(
    "gateway.providers.generation."
    "openai_compatible.requests.post"
)
def test_openai_compatible_generation(mock_post):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "id": "chat-1",
        "model": "granite",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello from Granite",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "total_tokens": 7,
        },
    }
    mock_post.return_value = response

    provider = OpenAICompatibleGenerationProvider(
        ProviderConfig(
            provider="vllm",
            model="granite",
            base_url="http://model:8000/v1",
            api_key="secret",
        )
    )

    result = provider.generate(
        GenerationRequest(
            messages=[
                GenerationMessage(
                    role="user",
                    content="Hello",
                )
            ]
        )
    )

    assert result.content == "Hello from Granite"
    assert result.provider == "vllm"
    assert result.usage["total_tokens"] == 7