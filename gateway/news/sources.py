"""
Trusted news source configuration.

Sources are RSS 2.0 feeds. Override the set via the NEWS_SOURCES
environment variable (JSON list of {name, url} objects).

Feed terms generally allow consuming headlines/summaries with
attribution — we store title + summary + link, never full article
text.
"""

import json
import logging
import os

logger = logging.getLogger("goalway.news")

DEFAULT_SOURCES = [
    {
        "name": "bbc-sport-football",
        "url": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    },
    {
        "name": "sky-sports-football",
        "url": "https://www.skysports.com/rss/12040",
    },
    {
        "name": "cnn-sport",
        "url": "http://rss.cnn.com/rss/edition_sport.rss",
    },
]


def get_sources() -> list[dict]:
    """Configured news sources (name + feed URL)."""
    raw = os.getenv("NEWS_SOURCES", "")

    if not raw:
        return list(DEFAULT_SOURCES)

    try:
        sources = json.loads(raw)

        if not isinstance(sources, list):
            raise ValueError("NEWS_SOURCES must be a JSON list")

        validated = []

        for source in sources:
            if not isinstance(source, dict) or not source.get("name") or not source.get("url"):
                raise ValueError(f"invalid source entry: {source}")

            validated.append(
                {"name": source["name"], "url": source["url"]}
            )

        return validated
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("invalid NEWS_SOURCES (%s), using defaults", exc)

        return list(DEFAULT_SOURCES)
