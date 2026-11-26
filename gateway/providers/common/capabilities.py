from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    model: str
    operations: frozenset[str]
    supports_batching: bool = False
    supports_streaming: bool = False
    supports_local_inference: bool = False
    max_batch_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, operation: str) -> bool:
        return operation in self.operations
    
    