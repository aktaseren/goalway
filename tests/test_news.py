"""Tests for trusted news ingestion (RSS -> knowledge base)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gateway.api as api_module
import gateway.news.ingest as news_ingest
from gateway.news.ingest import parse_feed, refresh_news
from gateway.news.sources import get_sources
from gateway.rag.models import Document, DocumentChunk, SearchResult

BBC_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>BBC Sport</title>
<item>
  <title>Arsenal win the London derby</title>
  <link>https://www.bbc.co.uk/sport/football/1</link>
  <guid isPermaLink="false">bbc-1</guid>
  <description>&lt;p&gt;Arsenal beat Chelsea 2-1 &amp;amp; move top.&lt;/p&gt;</description>
  <pubDate>Thu, 26 Nov 2026 18:03:00 GMT</pubDate>
</item>
<item>
  <title>Transfer window latest</title>
  <link>https://www.bbc.co.uk/sport/football/2</link>
  <guid isPermaLink="false">bbc-2</guid>
  <description>Clubs discuss a record striker transfer.</description>
  <pubDate>Wed, 25 Nov 2026 09:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

BROKEN_FEED = "<?xml version='1.0'?><rss><channel><title>broken"


def test_parse_feed_normalizes_items():
    items = parse_feed(BBC_FEED, "bbc-sport-football", max_items=10)

    assert len(items) == 2

    first = items[0]

    # HTML stripped, entities decoded
    assert first["title"] == "Arsenal win the London derby"
    assert first["description"] == "Arsenal beat Chelsea 2-1 & move top."
    assert first["source"] == "bbc-sport-football"
    assert first["guid"] == "bbc-1"
    assert first["published"] == "2026-11-26"
    assert first["year"] == 2026


def test_parse_feed_max_items():
    items = parse_feed(BBC_FEED, "bbc-sport-football", max_items=1)

    assert len(items) == 1


class FakeStoreRAGService:
    """Ingests into an in-memory store; get_document reflects it."""

    def __init__(self):
        self.documents: dict[str, list] = {}

    def get_document(self, document_id):
        return self.documents.get(document_id, [])

    def ingest(self, document: Document) -> int:
        chunks = [
            DocumentChunk(
                chunk_id=f"{document.document_id}-{i}",
                document_id=document.document_id,
                text=document.text,
                position=i,
                metadata=document.metadata,
            )
            for i in range(1)
        ]
        self.documents[document.document_id] = chunks

        return len(chunks)


def _mock_fetch(monkeypatch, xml_by_url):
    def fake_fetch(source, max_items=None):
        xml = xml_by_url.get(source["url"])

        if xml is None:
            raise RuntimeError("feed unreachable")

        return parse_feed(xml, source["name"], max_items or 10)

    monkeypatch.setattr(news_ingest, "fetch_source", fake_fetch)


def test_refresh_news_ingests_and_dedupes(monkeypatch):
    service = FakeStoreRAGService()
    monkeypatch.setattr(api_module, "rag_service", service)

    _mock_fetch(
        monkeypatch,
        {"https://feeds.bbci.co.uk/sport/football/rss.xml": BBC_FEED},
    )

    # only the BBC source configured
    monkeypatch.setenv("NEWS_SOURCES", '[{"name": "bbc-sport-football", "url": "https://feeds.bbci.co.uk/sport/football/rss.xml"}]')

    report = refresh_news()

    assert report["bbc-sport-football"] == {
        "fetched": 2,
        "ingested": 2,
        "skipped": 0,
    }
    assert len(service.documents) == 2

    # second fetch: same items -> all skipped, no duplicates
    report = refresh_news()

    assert report["bbc-sport-football"] == {
        "fetched": 2,
        "ingested": 0,
        "skipped": 2,
    }
    assert len(service.documents) == 2

    # ingested document carries source metadata for citations
    doc_id = next(iter(service.documents))
    metadata = service.documents[doc_id][0].metadata

    assert metadata["source"] == "bbc-sport-football"
    assert metadata["year"] == 2026
    assert metadata["published"]


def test_refresh_news_reports_source_errors(monkeypatch):
    service = FakeStoreRAGService()
    monkeypatch.setattr(api_module, "rag_service", service)

    _mock_fetch(monkeypatch, {})

    monkeypatch.setenv("NEWS_SOURCES", '[{"name": "bbc-sport-football", "url": "https://x/feed"}]')

    report = refresh_news()

    assert "error" in report["bbc-sport-football"]
    assert service.documents == {}


def test_news_fetch_tool_registered():
    from gateway.tools import get_tool_registry

    tool = get_tool_registry().get("news_fetch")

    assert tool is not None
    assert tool.required_permission == "rag.ingest"

    schema = tool.to_mcp()

    assert schema["name"] == "news_fetch"
    assert "sources" in schema["inputSchema"]["properties"]


def test_default_sources_configured():
    names = [source["name"] for source in get_sources()]

    assert "bbc-sport-football" in names
    assert "sky-sports-football" in names
    assert "cnn-sport" in names
    assert "trt-spor" in names
