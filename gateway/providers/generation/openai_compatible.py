import requests

from gateway.providers.common import (
    ProviderCapabilities,
    ProviderConfig,
)
from gateway.providers.generation.base import GenerationProvider
from gateway.providers.generation.models import (
    GenerationRequest,
    GenerationResponse,
)


class OpenAICompatibleGenerationProvider(GenerationProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = (
                f"Bearer {self.config.api_key}"
            )

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "temperature": request.temperature,
            "stream": False,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        payload.update(request.extra)

        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices")

        if not choices:
            raise ValueError(
                "Generation response did not contain choices"
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content")

        if not isinstance(content, str):
            raise ValueError(
                "Generation response did not contain message content"
            )

        usage = data.get("usage") or {}

        return GenerationResponse(
            content=content,
            model=data.get("model", self.config.model),
            provider=self.config.provider,
            finish_reason=choice.get("finish_reason"),
            usage={
                key: int(value)
                for key, value in usage.items()
                if isinstance(value, int)
            },
            metadata={
                "id": data.get("id"),
            },
        )

    def health(self) -> bool:
        try:
            response = requests.get(
                f"{self.config.base_url.rstrip('/')}/models",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.config.provider,
            model=self.config.model,
            operations=frozenset(
                {
                    "chat",
                    "completion",
                }
            ),
            supports_batching=False,
            supports_streaming=True,
            supports_local_inference=(
                self.config.provider in {
                    "vllm",
                    "openshift_ai",
                }
            ),
            max_batch_size=None,
        )