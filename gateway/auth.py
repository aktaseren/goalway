"""
Multi-provider authentication .

Resolution order:
1. Authorization: Bearer <token> -> OIDC provider (Keycloak)
2. X-API-Key header -> API key provider (key store w/ roles + quotas)

Providers are controlled by AUTH_PROVIDERS, e.g. "oidc,api_key".
A bearer token is only tried when the OIDC provider is enabled;
otherwise it falls through to the API key.
"""

from fastapi import Header, Request

from gateway.auth_context import AuthContext
from gateway.auth_providers.factory import create_auth_provider
from gateway.config import AUTH_PROVIDERS

# force provider registration
import gateway.auth_providers.oidc  # noqa: F401


def _bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization")

    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        return None

    return token.strip()


def get_auth_context(
        request: Request,
        x_api_key: str | None = Header(default=None),
) -> AuthContext:

    # 1. Bearer token -> OIDC (when enabled)
    if "oidc" in AUTH_PROVIDERS:
        token = _bearer_token(request)

        if token:
            provider = create_auth_provider("oidc")
            return provider.authenticate(
                request=request,
                credentials=token,
            )

    # 2. API key header
    if x_api_key:
        provider = create_auth_provider("api_key")
        return provider.authenticate(
            request=request,
            credentials=x_api_key,
        )

    from fastapi import HTTPException

    raise HTTPException(
        status_code=401,
        detail={
            "error": "Not authenticated",
            "hint": "Send X-API-Key or Authorization: Bearer <token>",
        },
    )
