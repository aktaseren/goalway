from typing import Any

import requests

from gateway.providers.tei.config import TEIConfig


class TEIClient:
    def __init__(self, config: TEIConfig):
        self.config = config

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        if self.config.embedding.api_key:
            headers["Authorization"] = (
                f"Bearer {self.config.embedding.api_key}"
            )

        return headers

    def health(self) -> bool:
        try:
            response = requests.get(
                self.config.health_url,
                headers=self._headers(),
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def info(self) -> dict[str, Any]:
        response = requests.get(
            self.config.info_url,
            headers=self._headers(),
            timeout=self.config.embedding.timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError(
                "TEI info response must be a JSON object"
            )

        return payload