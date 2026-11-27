"""
Trusted news ingestion (RSS -> knowledge base).

Fetches RSS 2.0 feeds from configured trusted sources (BBC Sport,
Sky Sports, CNN Sport by default) and ingests each item as a
document tagged with its source and publication date. Documents are
identified by a stable hash of the feed item's guid, so repeated
fetches never duplicate — only genuinely new items are ingested.

Every other layer benefits automatically: citations carry the
source, the recency horizon reflects real publication dates, and
the "No info found for <year>" warning only fires when the news
feeds themselves are behind.
"""

import hashlib
import html
import logging
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

from gateway.config import (
    NEWS_MAX_ITEMS,
    NEWS_TIMEOUT_SECONDS,
)
from gateway.news.sources import get_sources

logger = logging.getLogger("goalway.news")

USER_AGENT = "goalway/1.0 (+news ingestion demo)"


def _strip_html(text: str | None) -> str:
    """Feed descriptions embed HTML — reduce to plain text."""
    text = re.sub(r"<[^>]+>", " ", text or "")

    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _parse_date(pub_date: str | None) -> tuple[str, int] | tuple[None, None]:
    """RFC 822 pubDate -> (iso date, year)."""
    if not pub_date:
        return None, None

    try:
        parsed = parsedate_to_datetime(pub_date)

        return parsed.date().isoformat(), parsed.year
    except (TypeError, ValueError):
        return None, None


def parse_feed(xml_text: str, source_name: str, max_items: int) -> list[dict]:
    """Parse RSS 2.0 XML into normalized feed items."""
    root = ET.fromstring(xml_text)

    items = []

    for element in root.iter("item"):
        title = _strip_html(element.findtext("title", ""))
        link = (element.findtext("link", "") or "").strip()
        description = _strip_html(element.findtext("description", ""))
        pub_date = element.findtext("pubDate", "")
        guid = (element.findtext("guid", "") or link).strip()

        if not guid or not (title or description):
            continue

        published, year = _parse_date(pub_date)

        items.append(
            {
                "source": source_name,
                "guid": guid,
                "title": title,
                "description": description,
                "link": link,
                "published": published,
                "year": year,
            }
        )

        if len(items) >= max_items:
            break

    return items


def fetch_source(source: dict, max_items: int | None = None) -> list[dict]:
    """Fetch and parse one configured source."""
    response = requests.get(
        source["url"],
        timeout=NEWS_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    return parse_feed(
        response.text,
        source["name"],
        max_items or NEWS_MAX_ITEMS,
    )


def _document_id(source_name: str, guid: str) -> str:
    digest = hashlib.sha1(guid.encode("utf-8")).hexdigest()[:12]

    return f"{source_name}-{digest}"


def refresh_news(
    source_names: list[str] | None = None,
    max_items: int | None = None,
) -> dict:
    """
    Fetch all (or selected) sources and ingest new items.

    Returns a per-source report:
        {name: {fetched, ingested, skipped}} or {name: {error: ...}}
    """
    # Lazy import: gateway.api pulls in the tools package which
    # registers the news_fetch tool, which imports this module.
    from gateway.api import rag_service
    from gateway.rag.models import Document

    report: dict = {}

    for source in get_sources():
        name = source["name"]

        if source_names and name not in source_names:
            continue

        try:
            items = fetch_source(source, max_items)
        except Exception as exc:
            logger.warning("news fetch failed for %s: %s", name, exc)
            report[name] = {"error": str(exc)}
            continue

        ingested = 0
        skipped = 0

        for item in items:
            document_id = _document_id(name, item["guid"])

            # News items are immutable once published: skip anything
            # already in the knowledge base.
            if rag_service.get_document(document_id):
                skipped += 1
                continue

            text = ". ".join(
                part for part in (item["title"], item["description"]) if part
            )

            if not text:
                skipped += 1
                continue

            metadata = {
                "source": name,
                "link": item["link"],
                "published": item["published"],
                "category": "news",
            }

            if item["year"]:
                metadata["year"] = item["year"]

            rag_service.ingest(
                Document(
                    document_id=document_id,
                    text=text,
                    metadata=metadata,
                )
            )
            ingested += 1

        report[name] = {
            "fetched": len(items),
            "ingested": ingested,
            "skipped": skipped,
        }

        logger.info(
            "news refresh %s: %d fetched, %d new, %d existing",
            name,
            len(items),
            ingested,
            skipped,
        )

    return report
