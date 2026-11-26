from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key:str | None = None
    timeout_seconds: int = 120
    batch_size: int =32
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("Provider name cannot be empty")
        if not self.model.strip():
            raise ValueError("Provider model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("Provider timeout must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("Provider batch size must be greater than zero.")



        
        
