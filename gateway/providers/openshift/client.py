import requests

from gateway.providers.common import ProviderConfig
from gateway.providers.openshift.models import OpenShiftModel


class OpenShiftAIClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout_seconds: int = 30,
    ):
        if not endpoint.strip():
            raise ValueError("OpenShift AI endpoint cannot be empty")

        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        return headers

    def health(self) -> bool:
        try:
            response = requests.get(
                f"{self.endpoint}/models",
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[OpenShiftModel]:
        response = requests.get(
            f"{self.endpoint}/models",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        items = payload.get("data")

        if not isinstance(items, list):
            raise ValueError(
                "OpenShift AI response did not contain a data list"
            )

        models: list[OpenShiftModel] = []

        for item in items:
            model_name = item.get("id")

            if not isinstance(model_name, str):
                continue

            models.append(
                OpenShiftModel(
                    name=model_name,
                    url=self.endpoint,
                    ready=True,
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key != "id"
                    },
                )
            )

        return models

    def get_model(
        self,
        model_name: str,
    ) -> OpenShiftModel:
        for model in self.list_models():
            if model.name == model_name:
                return model

        raise ValueError(
            f"OpenShift AI model not found: {model_name}"
        )

    def create_generation_config(
        self,
        model_name: str,
    ) -> ProviderConfig:
        model = self.get_model(model_name)

        return ProviderConfig(
            provider="openshift_ai",
            model=model.name,
            base_url=model.url,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            batch_size=1,
            extra={
                "discovered": True,
                **model.metadata,
            },
        )