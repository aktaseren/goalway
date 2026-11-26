from gateway.rag.embeddings.health_manager import(
    EmbeddingHealthManager,
)

class EmbeddingProviderSelector:
    def __init__(
            self,
            health_manager: EmbeddingHealthManager,
    ):
        self.health_manager = health_manager

    def select(self) -> str:
        healthy = self.health_manager.healthy_providers()

        if not healthy:
            raise RuntimeError(
                "No healthy embedding providers available."
            )
        
        return healthy[0]