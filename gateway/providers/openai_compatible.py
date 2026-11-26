from gateway.providers.base import Provider
from gateway.providers.registry import register_provider

class OpenAICompatibleProvider(Provider):
    def build_payload(self, req):
        return req.model_dump()
    def parse_response(self, data):
        return data
    
register_provider("openai_compatible",
                  OpenAICompatibleProvider)