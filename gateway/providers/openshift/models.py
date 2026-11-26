from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpenShiftModel:
    name: str
    url: str
    ready: bool
    metadata: dict[str, Any] = field(default_factory=dict)