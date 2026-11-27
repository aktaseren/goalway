"""
Trusted news tools (RSS sources -> knowledge base).

Exposed through the tool registry: callable via MCP (tools/call
news_fetch), via OpenAI-style tool calling in chat, and via the
POST /v1/news/refresh endpoint.
"""

from gateway.tools.registry import tool


@tool(
    name="news_fetch",
    description=(
        "Fetch the latest news from trusted sources (BBC Sport, Sky "
        "Sports, CNN Sport) and add new items to the knowledge base. "
        "Already-known items are skipped. Returns a per-source report."
    ),
    parameters={
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of source names to fetch (e.g. ['bbc-sport-football'])",
            },
            "max_items": {
                "type": "integer",
                "description": "Max items per source (default 10)",
                "minimum": 1,
                "maximum": 50,
            },
        },
    },
    required_permission="rag.ingest",
    read_only=False,
)
def news_fetch(sources=None, max_items=None):
    from gateway.news.ingest import refresh_news

    source_names = sources if isinstance(sources, list) else None

    return refresh_news(
        source_names=source_names,
        max_items=max_items,
    )
