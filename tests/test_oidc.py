"""
Tests for gateway/auth_providers/oidc.py (bearer token auth).

Creates a real RSA keypair, signs tokens with PyJWT and serves the
matching JWKS via a mocked requests.get -- fully offline.
"""

import json
import os
import sys
import time
from unittest.mock import MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

import gateway.auth
import gateway.auth_providers.oidc as oidc_module
from gateway.auth_providers.oidc import OIDCProvider

ISSUER = "https://kc.test/realms/gateway"
AUDIENCE = "ai-gateway"
KID = "test-signing-key-1"

# One keypair for the whole module (generation is not free).
_private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)
_PRIVATE_PEM = _private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)


def _jwk() -> dict:
    from jwt.algorithms import RSAAlgorithm

    jwk = json.loads(RSAAlgorithm.to_jwk(_private_key.public_key()))
    jwk["kid"] = KID
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    return jwk


@pytest.fixture
def oidc_env(monkeypatch):
    """Configure the OIDC provider module + mocked JWKS endpoint."""
    monkeypatch.setattr(oidc_module, "OIDC_ISSUER", ISSUER)
    monkeypatch.setattr(oidc_module, "OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(oidc_module, "OIDC_JWKS_URL", "")
    monkeypatch.setattr(oidc_module, "OIDC_ADMIN_ROLES", {"admin"})
    monkeypatch.setattr(oidc_module, "OIDC_ROLE_CLAIM", "realm_access.roles")
    monkeypatch.setattr(
        oidc_module, "OIDC_USERNAME_CLAIM", "preferred_username"
    )

    # Fresh JWKS cache so every test fetches from the mocked endpoint.
    monkeypatch.setattr(
        oidc_module,
        "_JWKS_CACHE",
        {"url": None, "keys": [], "fetched_at": 0.0},
    )

    jwks = {"keys": [_jwk()]}

    def fake_get(url, timeout=None, **kwargs):
        assert url == f"{ISSUER}/.well-known/jwks.json"
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = jwks
        return response

    monkeypatch.setattr(oidc_module.requests, "get", fake_get)

    return oidc_module


def make_token(
    roles=("admin",),
    username="alice",
    issuer=ISSUER,
    audience=AUDIENCE,
    expires_in=3600,
    kid=KID,
):
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "exp": now + expires_in,
        "iat": now,
        "preferred_username": username,
        "realm_access": {"roles": list(roles)},
    }

    return jwt.encode(
        claims, _PRIVATE_PEM, algorithm="RS256", headers={"kid": kid}
    )


def fake_request():
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


def authenticate(token: str):
    provider = OIDCProvider()
    return provider.authenticate(request=fake_request(), credentials=token)


def test_valid_token_with_admin_role(oidc_env):
    context = authenticate(make_token(roles=["admin"]))

    assert context.authenticated is True
    assert context.auth_type == "oidc"
    assert context.role == "admin"
    assert context.username == "alice"
    assert context.client_ip == "127.0.0.1"


def test_valid_token_with_user_role(oidc_env):
    context = authenticate(make_token(roles=["basic-user"]))

    assert context.role == "user"
    assert context.username == "alice"


def test_valid_token_flat_role_claim(oidc_env, monkeypatch):
    monkeypatch.setattr(oidc_env, "OIDC_ROLE_CLAIM", "role")

    now = int(time.time())
    token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": now + 3600,
            "iat": now,
            "preferred_username": "bob",
            "role": "admin",
        },
        _PRIVATE_PEM,
        algorithm="RS256",
        headers={"kid": KID},
    )

    assert authenticate(token).role == "admin"


def test_expired_token_rejected(oidc_env):
    token = make_token(expires_in=-100)

    with pytest.raises(HTTPException) as excinfo:
        authenticate(token)

    assert excinfo.value.status_code == 401


def test_wrong_issuer_rejected(oidc_env):
    token = make_token(issuer="https://evil.example.com/realms/other")

    with pytest.raises(HTTPException) as excinfo:
        authenticate(token)

    assert excinfo.value.status_code == 401


def test_wrong_audience_rejected(oidc_env):
    token = make_token(audience="some-other-api")

    with pytest.raises(HTTPException) as excinfo:
        authenticate(token)

    assert excinfo.value.status_code == 401


def test_missing_kid_rejected(oidc_env):
    token = make_token(kid="unknown-key-id")

    with pytest.raises(HTTPException) as excinfo:
        authenticate(token)

    assert excinfo.value.status_code == 401


def test_malformed_token_rejected(oidc_env):
    with pytest.raises(HTTPException) as excinfo:
        authenticate("not-a-real-jwt")

    assert excinfo.value.status_code == 401


def test_empty_credentials_rejected(oidc_env):
    with pytest.raises(HTTPException) as excinfo:
        authenticate("")

    assert excinfo.value.status_code == 401


def test_get_auth_context_bearer_token(oidc_env, monkeypatch):
    monkeypatch.setattr(
        gateway.auth, "AUTH_PROVIDERS", ["oidc", "api_key"]
    )

    request = fake_request()
    request.headers = {
        "authorization": f"Bearer {make_token(roles=['admin'])}"
    }

    context = gateway.auth.get_auth_context(request, x_api_key=None)

    assert context.auth_type == "oidc"
    assert context.role == "admin"


def test_get_auth_context_skips_bearer_when_oidc_disabled(
    oidc_env, monkeypatch
):
    monkeypatch.setattr(gateway.auth, "AUTH_PROVIDERS", ["api_key"])

    request = fake_request()
    request.headers = {
        "authorization": "Bearer some-oidc-token",
        "x-api-key": "redhat-demo-key",
    }

    context = gateway.auth.get_auth_context(
        request, x_api_key="redhat-demo-key"
    )

    # Fell through to the API key provider, ignoring the bearer token.
    assert context.auth_type == "api_key"
    assert context.role == "admin"


def test_get_auth_context_no_credentials(monkeypatch):
    monkeypatch.setattr(gateway.auth, "AUTH_PROVIDERS", ["api_key"])

    request = fake_request()
    request.headers = {}

    with pytest.raises(HTTPException) as excinfo:
        gateway.auth.get_auth_context(request, x_api_key=None)

    assert excinfo.value.status_code == 401
