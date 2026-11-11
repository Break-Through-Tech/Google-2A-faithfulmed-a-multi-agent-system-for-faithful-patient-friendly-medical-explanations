"""Direct Gemini Verifier/Refiner loop for standalone use."""

from collections.abc import Awaitable, Callable
from typing import Any

from agents.model_runner import AgentDefinition, execute_agent
from agents.refiner import refiner
from agents.verifier import (
    check_verifier_output,
    make_verifier_input,
    verifier_agent,
)


MAX_ITERATIONS = 2
AgentExecutor = Callable[
    [AgentDefinition, dict[str, Any], str],
    Awaitable[dict[str, Any]],
]


async def run_refinement_loop(
    original_text: str,
    simplified_text: str,
    clinical_atoms: list[dict[str, Any]] | None = None,
    target_grade_level: int = 8,
    *,
    agent_executor: AgentExecutor = execute_agent,
) -> dict[str, Any]:
    """Verify a candidate, refine at most twice, and re-verify each revision."""
    candidate = simplified_text.strip()
    if not original_text.strip() or not candidate:
        raise ValueError("Original and simplified text must not be empty.")

    feedback: dict[str, Any] | None = None
    iterations = 0
    while True:
        verifier_input = make_verifier_input(
            original_text=original_text,
            simplified_text=candidate,
            clinical_atoms=clinical_atoms or [],
            target_grade_level=target_grade_level,
        )
        result = await agent_executor(
            verifier_agent,
            {
                "original_text": original_text,
                "simplified_text": candidate,
                "clinical_atoms": clinical_atoms or [],
                "target_grade_level": target_grade_level,
            },
            verifier_input,
        )
        raw_feedback = result.get("verifier_feedback")
        if not isinstance(raw_feedback, dict):
            raise TypeError("Verifier output must be a mapping.")
        feedback = check_verifier_output(raw_feedback).model_dump(mode="json")
        if feedback["overall_pass"] or iterations >= MAX_ITERATIONS:
            break

        refined = await agent_executor(
            refiner,
            {
                "original_text": original_text,
                "simplified_text": candidate,
                "verifier_feedback": feedback,
            },
            "Refine the simplified medical explanation.",
        )
        next_candidate = refined.get("simplified_text")
        if not isinstance(next_candidate, str) or not next_candidate.strip():
            raise RuntimeError("Refiner did not produce non-empty simplified text.")
        candidate = next_candidate.strip()
        iterations += 1

    assert feedback is not None
    return {
        "simplified_text": candidate,
        "verifier_feedback": feedback,
        "overall_pass": feedback["overall_pass"],
        "iterations": iterations,
    }
