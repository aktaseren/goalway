# Security Policy

## Supported versions

Only the latest `main` branch receives security fixes.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub
issues.**

Instead, open a [GitHub security advisory](https://github.com/aktaseren/goalway/security/advisories/new)
for this repository. Include:

- a description of the issue and its impact
- steps or a proof of concept to reproduce it
- affected components (gateway endpoints, MCP server, news ingestion, ...)

You should receive a response within a few days.

## Scope notes

This project is a demonstration platform. The default deployment ships
**demo API keys** (`redhat-demo-key` and friends) that are intentionally
public — treat them as such and rotate them before any real use. For
production-style deployments:

- set your own `API_KEYS_FILE` key store
- enable OIDC (`AUTH_PROVIDERS=oidc,api_key`) instead of relying on
  API keys alone
- mount tool credentials (`GITHUB_TOKEN`, `ARGOCD_TOKEN`) from Secrets,
  never bake them into images
