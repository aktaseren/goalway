"""
Multi-agent orchestration .

A four-role pipeline for complex tasks:

    Planner     splits the objective into concrete steps
    Researcher  gathers knowledge via RAG for each step
    Executor    produces the answer for each step, with tool access
    Reviewer    checks the final answer for groundedness and quality

Each role is backed by a model from the gateway registry (so
`auto` routing, health checks and failover all apply) and every
step records latency + token-ish sizes for observability.

The orchestrator is deliberately model-agnostic: every LLM
interaction goes through gateway.llm, and every knowledge lookup
through the RAG service -- no new infrastructure.
"""

import json
import logging
import time
import uuid

from gateway.config import (
    AGENT_EXECUTOR_MODEL,
    AGENT_PLANNER_MODEL,
    AGENT_REVIEWER_MODEL,
)
from gateway.llm import LLMError, complete_text
from gateway.metrics import AGENT_RUNS, AGENT_STEP_LATENCY
from gateway.rag.hallucination import groundedness_report

logger = logging.getLogger("goalway.agents")


PLANNER_SYSTEM = """You are a planning agent for an enterprise AI gateway.
Break the user's objective into 1-5 concrete, self-contained steps.
Respond with ONLY a JSON array of short step descriptions.
Example: ["Search the knowledge base for the API quota policy", "Summarize the policy in two sentences"]"""


RESEARCHER_SYSTEM = """You are a research agent. Given a step and retrieved
context, extract only the facts relevant to the step. Be concise and do not
invent anything that is not in the context."""


EXECUTOR_SYSTEM = """You are an executor agent in an enterprise AI gateway.
Complete the given step using the provided research notes. Be concise,
factual, and clearly flag anything you could not determine."""


REVIEWER_SYSTEM = """You are a reviewer agent. Judge whether the final answer
addresses the objective and is supported by the research notes.
Respond with ONLY a JSON object:
{"verdict": "approve" | "revise", "feedback": "<one sentence>"}"""


class Orchestrator:
    def __init__(
        self,
        planner_model: str = "",
        executor_model: str = "",
        reviewer_model: str = "",
        use_rag: bool = True,
    ):
        self.planner_model = planner_model or AGENT_PLANNER_MODEL
        self.executor_model = executor_model or AGENT_EXECUTOR_MODEL
        self.reviewer_model = reviewer_model or AGENT_REVIEWER_MODEL
        self.use_rag = use_rag

    # ------------------------------------------------------------------
    # Roles
    # ------------------------------------------------------------------

    def _plan(self, objective: str) -> list[str]:
        """Planner: objective -> ordered list of steps."""
        raw = complete_text(
            model=self.planner_model,
            prompt=objective,
            system=PLANNER_SYSTEM,
            temperature=0.0,
        )

        try:
            steps = json.loads(self._extract_json(raw))

            if isinstance(steps, list) and steps:
                return [str(step) for step in steps][:5]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: a single step covering the whole objective.
        return [objective]

    def _research(self, step: str, top_k: int = 3) -> tuple[str, list]:
        """Researcher: RAG lookup for a step. Returns (notes, results)."""
        if not self.use_rag:
            return "", []

        from gateway.api import rag_service

        results, _ = rag_service.search(query=step, top_k=top_k)

        if not results:
            return "No relevant documents found.", []

        notes = "\n\n".join(
            f"[{result.chunk.document_id}] {result.chunk.text}"
            for result in results
        )

        try:
            notes = complete_text(
                model=self.executor_model,
                prompt=(
                    f"Step: {step}\n\nContext:\n{notes[:4000]}\n\n"
                    "Extract only the facts relevant to the step."
                ),
                system=RESEARCHER_SYSTEM,
                temperature=0.0,
            )
        except LLMError:
            # Keep raw retrieval notes when no model is available.
            pass

        return notes, results

    def _execute(self, step: str, notes: str) -> str:
        """Executor: produce the answer for one step."""
        prompt = f"Step: {step}"

        if notes:
            prompt += f"\n\nResearch notes:\n{notes[:4000]}"

        return complete_text(
            model=self.executor_model,
            prompt=prompt,
            system=EXECUTOR_SYSTEM,
            temperature=0.2,
        )

    def _review(self, objective: str, answer: str, steps: list, results: list) -> dict:
        """Reviewer: groundedness check + model verdict."""
        # Check the step CONTENT for groundedness -- the formatted
        # answer's markdown headers are presentation, not claims.
        review_text = "\n\n".join(
            entry["answer"] for entry in steps
        ) or answer

        groundedness = groundedness_report(review_text, results)

        verdict = "approve"
        feedback = groundedness["summary"]

        if groundedness["verdict"] == "likely_hallucinated":
            verdict = "revise"
            feedback = (
                f"Answer is not grounded in retrieved context. "
                f"{feedback}"
            )

        # Model-based review on top, best effort.
        try:
            raw = complete_text(
                model=self.reviewer_model,
                prompt=(
                    f"Objective: {objective}\n\nAnswer:\n{answer[:3000]}"
                ),
                system=REVIEWER_SYSTEM,
                temperature=0.0,
            )

            parsed = json.loads(self._extract_json(raw))

            if isinstance(parsed, dict) and parsed.get("verdict") in (
                "approve",
                "revise",
            ):
                verdict = parsed["verdict"]
                feedback = str(parsed.get("feedback", feedback))[:300]
        except (LLMError, json.JSONDecodeError, ValueError):
            pass

        return {
            "verdict": verdict,
            "feedback": feedback,
            "groundedness": {
                "verdict": groundedness["verdict"],
                "score": groundedness["groundedness_score"],
                "supported_claims": groundedness["supported_claims"],
                "total_claims": groundedness["total_claims"],
            },
        }

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self, objective: str, max_steps: int = 5) -> "AgentRunResult":
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        started = time.time()

        AGENT_RUNS.inc()

        trace: list[dict] = []
        steps: list[dict] = []
        all_results: list = []

        # --- Planner ---------------------------------------------------
        planner_start = time.time()
        plan = self._plan(objective)[:max_steps]
        planner_ms = round((time.time() - planner_start) * 1000, 2)

        AGENT_STEP_LATENCY.labels(agent="planner").observe(planner_ms / 1000)

        trace.append(
            {
                "agent": "planner",
                "duration_ms": planner_ms,
                "steps_planned": len(plan),
            }
        )

        # --- Researcher + Executor per step ----------------------------
        for index, step in enumerate(plan, start=1):
            research_start = time.time()
            notes, results = self._research(step)
            research_ms = round((time.time() - research_start) * 1000, 2)

            AGENT_STEP_LATENCY.labels(agent="researcher").observe(
                research_ms / 1000
            )

            all_results.extend(results)

            execute_start = time.time()
            answer = self._execute(step, notes)
            execute_ms = round((time.time() - execute_start) * 1000, 2)

            AGENT_STEP_LATENCY.labels(agent="executor").observe(
                execute_ms / 1000
            )

            steps.append(
                {
                    "step": step,
                    "research_ms": research_ms,
                    "execute_ms": execute_ms,
                    "documents_used": [
                        result.chunk.document_id for result in results
                    ],
                    "answer": answer,
                }
            )

            trace.append(
                {
                    "agent": "researcher",
                    "step": index,
                    "duration_ms": research_ms,
                    "documents": len(results),
                }
            )
            trace.append(
                {
                    "agent": "executor",
                    "step": index,
                    "duration_ms": execute_ms,
                }
            )

        final_answer = "\n\n".join(
            f"**Step {index}: {entry['step']}**\n{entry['answer']}"
            for index, entry in enumerate(steps, start=1)
        )

        # --- Reviewer ---------------------------------------------------
        review_start = time.time()
        review = self._review(objective, final_answer, steps, all_results)
        review_ms = round((time.time() - review_start) * 1000, 2)

        AGENT_STEP_LATENCY.labels(agent="reviewer").observe(review_ms / 1000)

        trace.append(
            {
                "agent": "reviewer",
                "duration_ms": review_ms,
                "verdict": review["verdict"],
            }
        )

        total_ms = round((time.time() - started) * 1000, 2)

        logger.info(
            "agent_run run_id=%s steps=%d verdict=%s duration_ms=%s",
            run_id,
            len(steps),
            review["verdict"],
            total_ms,
        )

        return AgentRunResult(
            run_id=run_id,
            objective=objective,
            steps=steps,
            answer=final_answer,
            review=review,
            trace=trace,
            duration_ms=total_ms,
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip code fences around JSON."""
        text = text.strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]

        return text.strip()


class AgentRunResult:
    def __init__(
        self,
        run_id: str,
        objective: str,
        steps: list[dict],
        answer: str,
        review: dict,
        trace: list[dict],
        duration_ms: float,
    ):
        self.run_id = run_id
        self.objective = objective
        self.steps = steps
        self.answer = answer
        self.review = review
        self.trace = trace
        self.duration_ms = duration_ms

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "status": self.review["verdict"],
            "steps": self.steps,
            "answer": self.answer,
            "review": self.review,
            "trace": self.trace,
            "duration_ms": self.duration_ms,
        }
