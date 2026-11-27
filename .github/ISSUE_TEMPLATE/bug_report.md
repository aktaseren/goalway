name: Bug report
about: Something answered, ranked or declined incorrectly
labels: bug
body:
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: What you asked and what came back. If it's a chat/RAG issue, include the `gateway` metadata from the response (top_coverage, answer_coverage, citations) — it exists for debugging this.
      placeholder: |
        Asked: "tell me top 5 todays sport news"
        Got: 3 items instead of 5...
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: What did you expect?
    validations:
      required: true
  - type: dropdown
    id: deployment
    attributes:
      label: How are you running it?
      options:
        - docker compose
        - OpenShift (deploy.sh)
        - Helm chart
        - Local dev (uvicorn)
        - Other
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Logs / response metadata
      description: Pod or container logs, the full JSON response, screenshots — anything that shows the behavior.
      render: shell
