from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("Message role cannot be empty")

        if not self.content.strip():
            raise ValueError("Message content cannot be empty")


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[GenerationMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError(
                "Generation request must contain at least one message"
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "Generation temperature must be between 0 and 2"
            )

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(
                "Generation max_tokens must be greater than zero"
            )


@dataclass(frozen=True)
class GenerationResponse:
    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)