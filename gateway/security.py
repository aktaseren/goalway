from fastapi import HTTPException

from gateway.auth_context import AuthContext

def authenticate_api_key(
        provided_key: str | None,
        expected_key: str,
        client_ip: str,
) -> AuthContext:
    if provided_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid API key"}
        )
    
    return AuthContext(
        api_key=provided_key,
        client_ip=client_ip,
        authenticated= True,
    )

def is_prompt_safe(prompt: str) -> bool:
    blocked_terms = [
        "ignore previous instructions",
        "reveal system prompt",
        "bypass policy",
        "jailbreak",
    ]

    return not any(term in prompt.lower() for term in blocked_terms)