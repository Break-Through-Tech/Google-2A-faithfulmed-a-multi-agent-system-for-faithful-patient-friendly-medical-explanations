"""Execute a single agent step and hand its output to the next stage."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from agents.acceptance import assess_acceptance
from agents.contracts import PipelineCase
from agents.extractor_agent import (
    ExtractorOutput,
    extractor_agent,
    validate_extractor_output,
)
from agents.model_runner import AgentDefinition, execute_agent
from agents.readability.agent import root_agent as readability
from agents.readability.metrics import assess_candidate
from agents.refiner import refiner
from agents.simplifier.agent import root_agent as simplifier
from agents.verifier import (
    check_verifier_output,
    make_verifier_input,
    verifier_agent,
)


STAGES = ("Extractor", "Simplifier", "Readability", "Verifier", "Refiner")
MAX_REFINEMENTS = 2
AgentExecutor = Callable[
    [AgentDefinition, dict[str, Any], str],
    Awaitable[dict[str, Any]],
]


def _case_from_state(
    state: dict[str, Any],
    source_text: str,
    protected_facts: list[str],
    target_grade: int,
) -> PipelineCase:
    case_data = state.get("case")
    if not isinstance(case_data, dict):
        raise ValueError("Run the Extractor before running this agent.")
    case = PipelineCase.model_validate(case_data)
    if (
        case.source_text != source_text.strip()
        or case.protected_facts != protected_facts
        or case.target_grade != target_grade
    ):
        raise ValueError(
            "The source note or settings changed. Run the Extractor again first."
        )
    return case


def _clear_results(state: dict[str, Any], *keys: str) -> None:
    for key in keys:
        state.pop(key, None)


def _remove_stage_results(state: dict[str, Any], *stages: str) -> None:
    stage_results = state.get("stage_results", {})
    for stage in stages:
        stage_results.pop(stage, None)


async def run_agent_stage(
    stage: str,
    source_text: str,
    protected_facts: list[str],
    target_grade: int,
    previous_state: dict[str, Any] | None = None,
    *,
    use_retrieval: bool = False,
    agent_executor: AgentExecutor = execute_agent,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one named agent, record its visible input/output, and advance order."""
    if stage not in STAGES:
        raise ValueError(f"Unknown agent stage {stage!r}. Choose from {STAGES}.")

    state = dict(previous_state or {})
    stage_results = dict(state.get("stage_results", {}))
    state["stage_results"] = stage_results
    history = list(state.get("history", []))
    state["history"] = history

    if stage == "Extractor":
        if not source_text.strip():
            raise ValueError("Source clinical note cannot be empty.")
        _clear_results(
            state,
            "case",
            "extractor_output",
            "draft",
            "candidate",
            "verifier_feedback",
            "refinements",
            "acceptance",
            "readability_assessment",
        )
        _remove_stage_results(state, *STAGES)
        extractor_state = await agent_executor(
            extractor_agent,
            {"source_text": source_text},
            "Extract all source-grounded clinical facts from source_text.",
        )
        raw_output = extractor_state.get("extractor_output")
        if not isinstance(raw_output, (dict, ExtractorOutput)):
            raise TypeError("Extractor output must be a JSON object.")
        extraction = validate_extractor_output(raw_output, source_text)
        case = PipelineCase.from_extractor_output(
            source_text=source_text,
            extractor_output=extraction.model_dump(mode="json"),
            target_grade=target_grade,
            protected_facts=protected_facts,
        )
        state["extractor_output"] = extraction.model_dump(mode="json")
        state["case"] = case.model_dump(mode="json")
        state["refinements"] = 0
        input_data = {"source_text": source_text}
        output_data = state["extractor_output"]
        next_stage = "Simplifier"

    elif stage == "Simplifier":
        case = _case_from_state(
            state,
            source_text,
            protected_facts,
            target_grade,
        )
        _clear_results(
            state,
            "draft",
            "candidate",
            "verifier_feedback",
            "acceptance",
            "readability_assessment",
        )
        _remove_stage_results(state, "Readability", "Verifier", "Refiner")
        retrieval_result = {
            "status": "disabled",
            "message": "MedlinePlus retrieval is disabled for this run.",
            "passages": [],
        }
        if use_retrieval:
            from agents.retrieval import search_medlineplus

            terms = [
                atom.text
                for atom in case.atoms
                if atom.type
                in {"diagnosis", "symptom", "medication", "lab_value"}
            ]
            query = " ".join(terms) or case.source_text
            retrieval_result = search_medlineplus(query[:1000])
            passages = [
                {
                    "text": passage["text"],
                    "source": (
                        passage.get("url")
                        or passage.get("title")
                        or passage.get("source_type")
                        or "MedlinePlus"
                    ),
                }
                for passage in retrieval_result["passages"]
            ]
        else:
            passages = []
        case = PipelineCase.model_validate(
            {
                **case.model_dump(mode="json"),
                "retrieved_context": passages,
            }
        )
        state["case"] = case.model_dump(mode="json")
        input_data = case.to_state()
        result = await agent_executor(
            simplifier,
            input_data,
            "Create a patient-friendly explanation from the supplied source and atoms.",
        )
        draft = result.get("draft")
        if not isinstance(draft, str) or not draft.strip():
            raise RuntimeError("Simplifier did not produce non-empty 'draft' output.")
        state["draft"] = draft.strip()
        output_data = {
            "draft": state["draft"],
            "retrieval": retrieval_result,
        }
        next_stage = "Readability"

    elif stage == "Readability":
        case = _case_from_state(
            state,
            source_text,
            protected_facts,
            target_grade,
        )
        draft = state.get("draft")
        if not isinstance(draft, str) or not draft.strip():
            raise ValueError("Run the Simplifier before running Readability.")
        _clear_results(state, "candidate", "verifier_feedback", "acceptance")
        _clear_results(state, "readability_assessment")
        _remove_stage_results(state, "Verifier", "Refiner")
        input_data = {
            **case.to_state(),
            "draft": draft,
            "protected_facts": case.protected_facts,
        }
        result = await agent_executor(
            readability,
            input_data,
            "Improve the readability of the supplied patient-facing draft.",
        )
        candidate = result.get("readability_draft")
        if not isinstance(candidate, str) or not candidate.strip():
            raise RuntimeError(
                "Readability agent did not produce non-empty 'readability_draft'."
            )
        assessment = assess_candidate(
            draft,
            candidate.strip(),
            case.protected_facts,
            case.target_grade,
        )
        state["candidate"] = assessment.output_text
        state["readability_assessment"] = asdict(assessment)
        output_data = {
            "readability_draft": assessment.output_text,
            "assessment": state["readability_assessment"],
        }
        next_stage = "Verifier"

    elif stage == "Verifier":
        case = _case_from_state(
            state,
            source_text,
            protected_facts,
            target_grade,
        )
        candidate = state.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("Run Readability before running the Verifier.")
        verifier_input = make_verifier_input(
            original_text=case.source_text,
            simplified_text=candidate,
            clinical_atoms=case.verifier_atoms(),
            target_grade_level=case.target_grade,
        )
        input_data = json.loads(verifier_input)
        _remove_stage_results(state, "Refiner")
        result = await agent_executor(
            verifier_agent,
            {
                **case.to_state(),
                "original_text": case.source_text,
                "simplified_text": candidate,
                "clinical_atoms": case.verifier_atoms(),
            },
            verifier_input,
        )
        feedback = check_verifier_output(
            result.get("verifier_feedback", {})
        ).model_dump(mode="json")
        state["verifier_feedback"] = feedback
        state["acceptance"] = assess_acceptance(
            case.source_text,
            candidate,
            feedback,
            case.protected_facts,
        )
        output_data = {
            "verifier_feedback": feedback,
            "acceptance": state["acceptance"],
        }
        next_stage = (
            "Refiner"
            if not feedback["overall_pass"]
            and state.get("refinements", 0) < MAX_REFINEMENTS
            else None
        )

    else:
        case = _case_from_state(
            state,
            source_text,
            protected_facts,
            target_grade,
        )
        candidate = state.get("candidate")
        feedback = state.get("verifier_feedback")
        if not isinstance(candidate, str) or not isinstance(feedback, dict):
            raise ValueError("Run the Verifier and review its feedback before Refiner.")
        if feedback.get("overall_pass"):
            raise ValueError("Refiner is only available when Verifier reports issues.")
        refinements = int(state.get("refinements", 0))
        if refinements >= MAX_REFINEMENTS:
            raise ValueError(f"Refiner has reached the {MAX_REFINEMENTS}-run limit.")
        input_data = case.refiner_state(candidate, feedback)
        result = await agent_executor(
            refiner,
            input_data,
            "Refine the current explanation using the verifier feedback.",
        )
        revised = result.get("simplified_text")
        if not isinstance(revised, str) or not revised.strip():
            raise RuntimeError(
                "Refiner did not produce non-empty 'simplified_text' output."
            )
        state["candidate"] = revised.strip()
        state["refinements"] = refinements + 1
        _clear_results(state, "verifier_feedback", "acceptance")
        output_data = state["candidate"]
        next_stage = "Verifier"

    stage_result = {
        "agent": stage,
        "input": input_data,
        "output": output_data,
        "next_agent": next_stage,
    }
    stage_results[stage] = stage_result
    history.append(stage_result)
    state["next_stage"] = next_stage
    return state, stage_result
