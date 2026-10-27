"""Run the synthetic adversarial cases against the live ADK Verifier."""

import asyncio
import json
import uuid
from collections.abc import Mapping

from agents.adversarial_cases import ADVERSARIAL_CASES
from agents.run_pipeline import _as_mapping, execute_adk_agent
from agents.verifier import check_verifier_output, make_verifier_input, verifier_agent


async def evaluate_cases() -> dict[str, object]:
    run_id = uuid.uuid4().hex[:8]
    results = []
    for index, case in enumerate(ADVERSARIAL_CASES, start=1):
        state = {
            "original_text": case["source_text"],
            "simplified_text": case["candidate_text"],
            "clinical_atoms": [],
        }
        prompt = make_verifier_input(
            original_text=case["source_text"],
            simplified_text=case["candidate_text"],
        )
        agent_state = await execute_adk_agent(
            verifier_agent,
            f"faithfulmed_adversarial_{run_id}_{index}",
            state,
            prompt,
        )
        feedback = _as_mapping(agent_state.get("verifier_feedback"), "Verifier output")
        feedback = check_verifier_output(feedback).model_dump(mode="json")
        check = feedback.get(case["category"])
        if not isinstance(check, Mapping):
            raise RuntimeError(
                f"Verifier response omitted expected {case['category']} check for {case['name']}"
            )
        detected = check.get("passed") is False or bool(check.get("issues"))
        results.append({
            "name": case["name"],
            "expected_category": case["category"],
            "detected": detected,
            "feedback": feedback,
        })

    return {
        "cases": results,
        "detected_count": sum(result["detected"] for result in results),
        "case_count": len(results),
        "all_detected": all(result["detected"] for result in results),
        "note": "Synthetic calibration only; not a clinical safety certification.",
    }


async def _main() -> None:
    print(json.dumps(await evaluate_cases(), indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
