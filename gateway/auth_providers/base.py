from gateway.auth_context import AuthContext

class AuthProvider:
    def authenticate(self, request, credentials) -> AuthContext:
        raise NotImplementedError