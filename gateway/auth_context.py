from dataclasses import dataclass

@dataclass
class AuthContext:
    api_key: str
    client_ip: str
    authenticated: bool
    auth_type: str = "api_key"

    username: str = "admin"
    role: str = "admin"