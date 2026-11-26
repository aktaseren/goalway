from fastapi import HTTPException

from gateway.auth_context import AuthContext

ROLE_PERMISSIONS = {
    "admin": {
        "admin.reload",
        "chat.complete",
        "rag.ingest",
        "rag.search",
        "rag.evaluate",
        "rag.feedback",
        "agents.run",
        "tools.call",
        "mcp.use",
    },
    "user": {
        "chat.complete",
        "rag.search",
        "rag.feedback",
    },
}

def require_permission(
        auth: AuthContext,
        permission: str,
):
        permissions = ROLE_PERMISSIONS.get(auth.role, set())

        if permission not in permissions:
                raise HTTPException(
                        status_code=403,
                        detail = {
                                "error": "Permission denied",
                                "permission": permission,
                                "role": auth.role,
                        },
                )
