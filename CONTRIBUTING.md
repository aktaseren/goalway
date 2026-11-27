# Contributing to Goalway

Thanks for your interest in contributing! Goalway is an enterprise AI
gateway demonstrated on football news — every layer is meant to be
readable, testable and hackable.

## Getting started

```bash
git clone https://github.com/aktaseren/goalway.git
cd goalway

# development environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

# run the test suite (fully offline — all HTTP is mocked)
.venv/bin/python -m pytest tests/ -q

# run the whole stack locally
docker compose up --build
open http://localhost:8080/ui
```

## Finding your way around

```
gateway/
├── api.py            # HTTP endpoints; the rag chat flow lives here
├── main.py           # app assembly + startup tasks (health, news scheduler)
├── model_selection.py# model=auto routing
├── llm.py            # programmatic completions + tool-calling loop
├── rag/              # chunker, embeddings, retrievers, rerankers,
│   ├── qa.py         #   citations, groundedness, question shapes
│   └── news/         #   RSS ingestion from trusted sources
├── tools/            # tool registry + k8s/github/argocd/news tools
├── mcp/              # MCP server (JSON-RPC 2.0)
├── agents/           # planner/researcher/executor/reviewer pipeline
├── auth_providers/   # api_key + oidc (Keycloak/JWKS)
├── keys.py           # API key store: roles + per-key quotas
├── cache.py          # Redis response cache (degrades gracefully)
└── registry.py       # model registry + background health probing
static/index.html     # the demo console (one file, no build tooling)
deploy/               # OpenShift manifests, Helm chart, deploy.sh
cicd/                 # Tekton pipeline
```

## Design principles to preserve

- **Honest answers.** Wrong answers are worse than no answers: relevance
  gates decline rather than misquote, answers disclose their knowledge
  horizon, missing years are surfaced, not hidden.
- **Questions have shapes.** Single-answer, list and enumeration
  questions are answered differently — check `gateway/rag/qa.py` before
  changing answer behavior, and add a test for your question shape.
- **A cache is an optimization, never a dependency.** Cache failures
  degrade to misses; never let one fail a request.
- **One tool registry, two protocols.** New tools registered in
  `gateway/tools/` are instantly available to chat tool calling, MCP
  clients and agents — no extra wiring.
- **No build tooling.** The UI is a single static file; the stack runs
  on plain Python + Docker. Keep it that way unless the trade-off is
  overwhelming.

## Pull requests

1. Fork / branch from `main`.
2. **Add tests** for any behavior change — the suite is offline and
   fast (~25s), there is no excuse. 🙂
3. Keep PRs focused; one capability per PR.
4. Verify locally:
   ```bash
   .venv/bin/python -m pytest tests/ -q
   docker compose up --build   # then poke the UI
   ```
5. CI must pass: the suite runs on Python 3.12 + 3.13, both images
   build, and the composed stack is smoke-tested live.

### Adding a trusted news source

Sources are RSS feeds configured in `gateway/news/sources.py` (or the
`NEWS_SOURCES` env var). Only add feeds you trust; store just
title + summary + link, never full article text.

### Adding a tool

Register it in `gateway/tools/` with the `@tool` decorator: a JSON
schema for arguments, an RBAC permission, and a read-only flag. It is
then callable via chat `tools`, MCP `tools/call` and agents
automatically. Add a test in `tests/test_tools_registry.py` style.

## Reporting bugs

Open an issue with the bug template: what you asked, what came back,
what you expected. The gateway's response metadata (`top_coverage`,
`answer_coverage`, `citations`) is designed for debugging — include
it when you can.

## License

By contributing you agree that your contributions are licensed under
Apache-2.0.
