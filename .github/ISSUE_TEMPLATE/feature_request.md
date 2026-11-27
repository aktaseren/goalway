name: Feature request
about: Suggest a capability, source or question shape
labels: enhancement
body:
  - type: textarea
    id: problem
    attributes:
      label: What problem does this solve?
      description: A question the system answers wrongly or declines today, an operation that needs a tool, a source that should be ingested...
      placeholder: I asked "..." and the answer was wrong because...
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: What should happen instead?
    validations:
      required: false
  - type: dropdown
    id: area
    attributes:
      label: Which layer?
      options:
        - Chat / answer quality
        - RAG retrieval / ranking
        - Groundedness / evaluation
        - News ingestion / sources
        - MCP / tools
        - Agents
        - Auth / quotas
        - Deployment / ops
        - UI console
        - Other / not sure
    validations:
      required: true
