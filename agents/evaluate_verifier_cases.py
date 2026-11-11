"""Run the synthetic adversarial cases against the live Gemini Verifier."""

import asyncio
import json
from collections.abc import Mapping

from agents.adversarial_cases import ADVERSARIAL_CASES
from agents.model_runner import execute_agent
from agents.verifier import check_verifier_output, make_verifier_input, verifier_agent


async def evaluate_cases() -> dict[str, object]:
    results = []
    for case in ADVERSARIAL_CASES:
        state = {
            "original_text": case["source_text"],
            "simplified_text": case["candidate_text"],
            "clinical_atoms": [],
        }
        prompt = make_verifier_input(
            original_text=case["source_text"],
            simplified_text=case["candidate_text"],
        )
        agent_state = await execute_agent(
            verifier_agent,
            state,
            prompt,
        )
        feedback = agent_state.get("verifier_feedback")
        if not isinstance(feedback, Mapping):
            raise TypeError("Verifier output must be a mapping.")
        feedback = check_verifier_output(dict(feedback)).model_dump(mode="json")
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
