"""
RAG tools exposed to models and MCP clients .

These give an agent the ability to search and ingest knowledge
through the same RAG pipeline the HTTP API uses.
"""

from gateway.tools.registry import tool

_RAG_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return (default 5)",
            "minimum": 1,
            "maximum": 20,
        },
    },
    "required": ["query"],
}


@tool(
    name="rag_search",
    description=(
        "Search the knowledge base for documents relevant to a query. "
        "Returns chunks with document ids and text."
    ),
    parameters=_RAG_SEARCH_PARAMETERS,
    required_permission="rag.search",
)
def rag_search(query: str, top_k: int = 5):
    from gateway.api import rag_service

    results, stats = rag_service.search(query=query, top_k=top_k)

    return {
        "results": [
            {
                "document_id": result.chunk.document_id,
                "chunk_id": result.chunk.chunk_id,
                "text": result.chunk.text,
                "score": round(result.score, 4),
            }
            for result in results
        ],
        "total": len(results),
    }


@tool(
    name="rag_ingest",
    description=(
        "Add a document to the knowledge base so it can be searched. "
        "Provide a stable document id and the full text."
    ),
    parameters={
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "Stable identifier for the document",
            },
            "text": {
                "type": "string",
                "description": "Full document text",
            },
        },
        "required": ["document_id", "text"],
    },
    required_permission="rag.ingest",
    read_only=False,
)
def rag_ingest(document_id: str, text: str):
    from gateway.api import rag_service
    from gateway.rag.models import Document

    chunks = rag_service.ingest(
        Document(document_id=document_id, text=text)
    )

    return {
        "document_id": document_id,
        "chunks_created": chunks,
    }
