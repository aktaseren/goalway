from fastapi import HTTPException

from gateway.auth_context import AuthContext
from gateway.auth_providers.base import AuthProvider
from gateway.auth_providers.registry import register_auth_provider
from gateway.keys import lookup_key


class APIKeyProvider(AuthProvider):
    def authenticate(self, request, credentials):
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail={"error": "Missing API key"},
            )

        entry = lookup_key(credentials)

        if entry is None:
            raise HTTPException(
                status_code=401,
                detail={"error": "Invalid API key"},
            )

        return AuthContext(
            api_key=credentials,
            client_ip=request.client.host if request.client else "",
            authenticated=True,
            auth_type="api_key",
            username=entry["name"],
            role=entry["role"],
        )


register_auth_provider(
     "api_key",
     APIKeyProvider,
)
