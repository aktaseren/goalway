from pydantic import BaseModel

class ChatRequest(BaseModel):
    model: str
    messages: list
    conversation_id: str | None = None

    # OpenAI-style tool calling. When tools are provided the
    # gateway runs the agentic loop and returns the final answer.
    tools: list | None = None
    tool_choice: str | None = None


class AgentRunRequest(BaseModel):
    objective: str
    planner_model: str | None = None
    executor_model: str | None = None
    reviewer_model: str | None = None
    use_rag: bool = True
    max_steps: int = 5
