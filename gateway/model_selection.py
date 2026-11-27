"""
Intelligent model selection for model="auto" requests.

Chooses a backend model for a prompt when the client requests
model="auto". The demo theme is football news: domain terms
route to the strongest chat backend (granite), while greetings and
very short prompts go to the cheap echo backend.
"""

GREETING_TERMS = ["hi", "hello", "hey"]

TECHNICAL_TERMS = [
    "openshift",
    "kubernetes",
    "container",
    "pod",
    "route",
    "service",
    "deployment",
    "operator",
    "ai gateway",
    "model serving",
]

# demo: football news domain routing
FOOTBALL_TERMS = [
    "football",
    "soccer",
    "premier league",
    "la liga",
    "serie a",
    "bundesliga",
    "ligue 1",
    "champions league",
    "europa league",
    "world cup",
    "ballon d'or",
    "ballon",
    "transfer",
    "striker",
    "goalkeeper",
    "midfielder",
    "match report",
    "kickoff",
    "penalty",
    "goal",
    "goals",
    "clean sheet",
    "hat-trick",
    "hat trick",
    "aggregate",
    # news intent: questions about news belong to the knowledge base
    # model, not a chat backend
    "news",
    "headline",
    "headlines",
]


def choose_model(prompt: str) -> str:
    clean_prompt = prompt.strip()
    prompt_lower = clean_prompt.lower()

    if prompt_lower in GREETING_TERMS:
        return "echo"

    if len(clean_prompt) <= 10:
        return "echo"

    if any(term in prompt_lower for term in TECHNICAL_TERMS):
        return "granite"

    # Football questions go to the built-in RAG model: retrieval +
    # extractive answering with citations, no backend needed.
    if any(term in prompt_lower for term in FOOTBALL_TERMS):
        return "rag"

    return "tinyllama"
