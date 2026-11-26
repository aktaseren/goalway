from gateway.rag.models import Document
from gateway.rag.chunker import TextChunker
from gateway.rag.service import RAGService

service = RAGService(
    chunker=TextChunker(
        chunk_size=70,
        overlap=15,
    )
)

count = service.ingest(
    Document(
        document_id="1",
        text="""
OpenShift AI provides enterprise model serving.

Ollama serves embedding models.

Granite models are developed by IBM.
""",
        metadata={}
    )
)

print("Chunks:", count)

results, stats = service.search(
    "Which platform serves embedding models?",
    top_k=3
)

print()
print("Stats:", stats)
print()

for r in results:
    print(r.score)
    print(r.chunk.text)
    print()