#!/usr/bin/env python3
"""
Seed the gateway's knowledge base with football news demo.

    python3 examples/seed_football.py [gateway-url] [api-key]

Defaults: http://localhost:8080 and redhat-demo-key.
Uses only the standard library -- run it from anywhere.
"""

import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")
KEY = sys.argv[2] if len(sys.argv) > 2 else "redhat-demo-key"


def post(path, payload):
    request = urllib.request.Request(
        f"{URL}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-API-Key": KEY,
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def main():
    with open(os.path.join(HERE, "football_news.json")) as f:
        documents = json.load(f)["documents"]

    print(f"Seeding {len(documents)} football news documents into {URL}")

    total_chunks = 0

    for doc in documents:
        result = post(
            "/v1/rag/ingest",
            {
                "document_id": doc["document_id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
            },
        )
        total_chunks += result["chunks_created"]
        print(f"  ✓ {doc['document_id']} ({result['chunks_created']} chunks)")

    print(f"\nDone: {total_chunks} chunks ingested.")
    print(f"Try: curl -s {URL}/v1/rag/search -H 'X-API-Key: {KEY}' \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"query\": \"who won the champions league final\"}'")


if __name__ == "__main__":
    main()
