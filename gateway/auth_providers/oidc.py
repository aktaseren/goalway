"""
OIDC / Keycloak auth provider .

Validates JWT bearer tokens against the issuer's JWKS endpoint,
checks issuer and audience, and maps token roles to gateway roles
(admin / user).

Requires PyJWT + cryptography. Configure via:

    AUTH_PROVIDERS=oidc,api_key
    OIDC_ISSUER=https://keycloak.example.com/realms/gateway
    OIDC_AUDIENCE=goalway
"""

import logging
import time

import requests
from fastapi import HTTPException

from gateway.auth_context import AuthContext
from gateway.auth_providers.base import AuthProvider
from gateway.auth_providers.registry import register_auth_provider
from gateway.config import (
    OIDC_ADMIN_ROLES,
    OIDC_AUDIENCE,
    OIDC_ISSUER,
    OIDC_JWKS_URL,
    OIDC_ROLE_CLAIM,
    OIDC_USERNAME_CLAIM,
)

logger = logging.getLogger("goalway.oidc")

_JWKS_CACHE = {
    "url": None,
    "keys": [],
    "fetched_at": 0.0,
}

JWKS_CACHE_TTL_SECONDS = 300


class OIDCProvider(AuthProvider):
    def authenticate(self, request, credentials):
        """
        credentials is the raw Bearer token (without the scheme).
        """
        import jwt  # lazy: keeps api_key-only deployments dependency-free

        if not OIDC_ISSUER:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "OIDC provider is not configured",
                    "hint": "Set OIDC_ISSUER to enable bearer token auth",
                },
            )

        if not credentials:
            raise HTTPException(
                status_code=401,
                detail={"error": "Missing bearer token"},
            )

        try:
            header = jwt.get_unverified_header(credentials)
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=401,
                detail={"error": "Malformed token"},
            )

        signing_key = self._find_signing_key(header.get("kid", ""))

        if signing_key is None:
            raise HTTPException(
                status_code=401,
                detail={"error": "Token signing key not found"},
            )

        try:
            options = {"verify_aud": bool(OIDC_AUDIENCE)}
            claims = jwt.decode(
                credentials,
                signing_key,
                algorithms=["RS256", "RS384", "RS512", "ES256"],
                issuer=OIDC_ISSUER,
                audience=OIDC_AUDIENCE or None,
                options=options,
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail={"error": "Token expired"},
            )
        except jwt.InvalidIssuerError:
            raise HTTPException(
                status_code=401,
                detail={"error": "Invalid token issuer"},
            )
        except jwt.InvalidAudienceError:
            raise HTTPException(
                status_code=401,
                detail={"error": "Invalid token audience"},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Invalid token",
                    "details": str(exc),
                },
            )

        username = claims.get(OIDC_USERNAME_CLAIM) or "unknown"
        token_roles = self._extract_roles(claims)
        role = "admin" if token_roles & OIDC_ADMIN_ROLES else "user"

        return AuthContext(
            api_key="",
            client_ip=request.client.host if request.client else "",
            authenticated=True,
            auth_type="oidc",
            username=username,
            role=role,
        )

    @staticmethod
    def _extract_roles(claims: dict) -> set:
        """
        Walk the configured role claim. Supports nested claims like
        realm_access.roles (Keycloak) and flat lists.
        """
        value: object = claims
        for part in OIDC_ROLE_CLAIM.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return set()

        if isinstance(value, list):
            return {str(r) for r in value}
        if isinstance(value, str):
            return {value}

        return set()

    def _find_signing_key(self, kid: str):
        keys = self._fetch_jwks()

        for key in keys:
            if key.get("kid") == kid:
                try:
                    import jwt.algorithms

                    return jwt.algorithms.RSAAlgorithm.from_jwk(
                        __import__("json").dumps(key)
                    )
                except Exception as exc:
                    logger.warning("failed to load JWKS key %s: %s", kid, exc)
                    return None

        # Unknown kid: maybe the realm rotated keys, refresh once.
        keys = self._fetch_jwks(force=True)

        for key in keys:
            if key.get("kid") == kid:
                try:
                    import jwt.algorithms

                    return jwt.algorithms.RSAAlgorithm.from_jwk(
                        __import__("json").dumps(key)
                    )
                except Exception as exc:
                    logger.warning("failed to load JWKS key %s: %s", kid, exc)

        return None

    @staticmethod
    def _fetch_jwks(force: bool = False) -> list:
        global _JWKS_CACHE

        url = OIDC_JWKS_URL or f"{OIDC_ISSUER.rstrip('/')}/.well-known/jwks.json"

        if (
            not force
            and _JWKS_CACHE["url"] == url
            and _JWKS_CACHE["keys"]
            and time.time() - _JWKS_CACHE["fetched_at"] < JWKS_CACHE_TTL_SECONDS
        ):
            return _JWKS_CACHE["keys"]

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        keys = response.json().get("keys", [])

        _JWKS_CACHE = {
            "url": url,
            "keys": keys,
            "fetched_at": time.time(),
        }

        return keys


register_auth_provider(
    "oidc",
    OIDCProvider,
)
