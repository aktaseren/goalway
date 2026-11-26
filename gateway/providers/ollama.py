from gateway.providers.base import Provider
from gateway.providers.registry import register_provider

class OllamaProvider(Provider):
    def build_payload(self, req):
        prompt = req.messages[-1].get("content", "")

        return {
            "model": req.model,
            "prompt": prompt,
            "stream": False,
        }
    
    def parse_response(self, data):
        return{
            "model": data.get("model", "ollama"),
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": data.get("response", "")
                    }
                }
            ]
        }
    
register_provider(
    "ollama",
    OllamaProvider
)