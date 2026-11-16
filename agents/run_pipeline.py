"""Run the FaithfulMed agent handoffs with a bounded Verifier/Refiner loop."""

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pydantic import BaseModel

from agents.acceptance import assess_acceptance
from agents.contracts import PipelineCase
from agents.model_runner import AgentDefinition, execute_agent
from agents.readability.agent import root_agent as readability
from agents.refiner import refiner
from agents.simplifier.agent import root_agent as simplifier
from agents.verifier import (
    check_verifier_output,
    make_verifier_input,
    verifier_agent,
)


MAX_REFINEMENTS = 2
AgentExecutor = Callable[
    [AgentDefinition, dict[str, Any], str],
    Awaitable[dict[str, Any]],
]

SYNTHETIC_SOURCE = (
    "The patient reports a dry cough for three days. No fever or shortness of "
    "breath is reported. A viral upper respiratory infection is suspected. "
    "The plan is rest, fluids, and follow-up if symptoms worsen or persist."
)
SYNTHETIC_EXTRACTOR_OUTPUT = {
    "atoms": [
        {
            "text": "The patient reports a dry cough for three days",
            "type": "symptom",
            "source_quote": "reports a dry cough for three days",
        },
        {
            "text": "No fever is reported",
            "type": "negative_finding",
            "source_quote": "No fever",
        },
        {
            "text": "No shortness of breath is reported",
            "type": "negative_finding",
            "source_quote": "No fever or shortness of breath is reported",
        },
        {
            "text": "A viral upper respiratory infection is suspected",
            "type": "diagnosis",
            "source_quote": "A viral upper respiratory infection is suspected",
        },
        {
            "text": "The plan is rest and fluids",
            "type": "treatment",
            "source_quote": "The plan is rest, fluids",
        },
        {
            "text": "Follow up if symptoms worsen or persist",
            "type": "follow_up",
            "source_quote": "follow-up if symptoms worsen or persist",
        },
    ],
    "not_found_types": [
        "diagnosis_reasoning",
        "risk",
        "medical_history",
        "family_history",
        "environmental_history",
        "demographic",
        "medical_specialty",
        "examination",
        "examination_finding",
        "lab_value",
        "vital_sign",
        "procedure",
        "medication",
        "referral",
        "patient_education",
    ],
}


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        f"{label} must be a mapping or Pydantic model, got {type(value).__name__}"
    )


async def run_pipeline(
    case: PipelineCase,
    *,
    max_refinements: int = MAX_REFINEMENTS,
    exercise_refiner: bool = False,
    agent_executor: AgentExecutor = execute_agent,
) -> dict[str, Any]:
    """Run Simplifier -> Readability -> Verifier and bounded Refiner handoffs."""
    if max_refinements < 0:
        raise ValueError("max_refinements cannot be negative")

    base_state = case.to_state()
    simplifier_state = await agent_executor(
        simplifier,
        base_state,
        "Create a patient-friendly explanation from the supplied source and clinical atoms.",
    )
    draft = simplifier_state.get("draft")
    if not isinstance(draft, str) or not draft.strip():
        raise RuntimeError("Simplifier did not produce non-empty 'draft' output")
    draft = draft.strip()

    readability_state = await agent_executor(
        readability,
        {
            **base_state,
            "draft": draft,
            "protected_facts": case.protected_facts,
        },
        "Improve the readability of the supplied patient-facing draft.",
    )
    candidate = readability_state.get("readability_draft")
    if not isinstance(candidate, str) or not candidate.strip():
        raise RuntimeError(
            "Readability agent did not produce non-empty 'readability_draft'"
        )
    candidate = candidate.strip()
    readability_candidate = candidate
    if exercise_refiner:
        candidate = f"{candidate}\nThe suspected viral infection is confirmed."
        print(
            "[demo] Injected a false certainty claim to exercise Verifier and Refiner.",
            flush=True,
        )
    initial_candidate = candidate

    verifier_atoms = case.verifier_atoms()
    refinements = 0
    feedback: dict[str, Any] | None = None
    initial_feedback: dict[str, Any] | None = None
    initial_acceptance: dict[str, Any] | None = None
    acceptance: dict[str, Any] | None = None

    while True:
        verifier_input = make_verifier_input(
            original_text=case.source_text,
            simplified_text=candidate,
            clinical_atoms=verifier_atoms,
            target_grade_level=case.target_grade,
        )
        verifier_state = await agent_executor(
            verifier_agent,
            {
                **base_state,
                "original_text": case.source_text,
                "simplified_text": candidate,
                "clinical_atoms": verifier_atoms,
            },
            verifier_input,
        )
        feedback = _as_mapping(
            verifier_state.get("verifier_feedback"),
            "Verifier output",
        )
        feedback = check_verifier_output(feedback).model_dump(mode="json")
        acceptance = assess_acceptance(
            case.source_text,
            candidate,
            feedback,
            case.protected_facts,
        )
        if initial_feedback is None:
            initial_feedback = dict(feedback)
            initial_acceptance = acceptance

        if feedback["overall_pass"] or refinements >= max_refinements:
            break

        feedback_for_refiner = {**feedback, "overall_pass": False}
        refiner_state = await agent_executor(
            refiner,
            case.refiner_state(candidate, feedback_for_refiner),
            "Refine the current explanation using the verifier feedback.",
        )
        refined = refiner_state.get("simplified_text")
        if not isinstance(refined, str) or not refined.strip():
            raise RuntimeError(
                "Refiner did not produce non-empty 'simplified_text' output"
            )
        candidate = refined.strip()
        refinements += 1

    assert feedback is not None
    assert acceptance is not None
    automated_checks_passed = (
        feedback["overall_pass"] and acceptance["deterministic_checks_passed"]
    )
    return {
        "draft": draft,
        "readability_draft": readability_candidate,
        "initial_candidate": initial_candidate,
        "refiner_demo": exercise_refiner,
        "refiner_exercised": refinements > 0,
        "initial_verifier_feedback": initial_feedback,
        "initial_acceptance": initial_acceptance,
        "final_text": candidate,
        "verifier_feedback": feedback,
        "acceptance": acceptance,
        "automated_checks_passed": automated_checks_passed,
        "overall_pass": automated_checks_passed,
        "review_status": "human_review_required",
        "refinements": refinements,
        "max_refinements": max_refinements,
    }


def build_synthetic_case() -> PipelineCase:
    """Create the synthetic demo case using the Extractor notebook output shape."""
    return PipelineCase.from_extractor_output(
        source_text=SYNTHETIC_SOURCE,
        extractor_output=SYNTHETIC_EXTRACTOR_OUTPUT,
        target_grade=8,
    )


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=MAX_REFINEMENTS,
        help="Maximum number of Refiner calls (default: 2)",
    )
    parser.add_argument(
        "--exercise-refiner",
        action="store_true",
        help=(
            "Append a deliberately unsupported confirmed-diagnosis claim to the "
            "synthetic draft to exercise Verifier/Refiner"
        ),
    )
    args = parser.parse_args()
    result = await run_pipeline(
        build_synthetic_case(),
        max_refinements=args.max_refinements,
        exercise_refiner=args.exercise_refiner,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
