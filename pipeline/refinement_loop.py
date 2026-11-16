import json

from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.verifier import verifier_agent
from agents.refiner import refiner


MAX_ITERATIONS = 2


async def run_refinement_loop(
    original_text,
    simplified_text,
    clinical_atoms=None,
    target_grade_level=8,
):
    verifier_runner = InMemoryRunner(
        agent=verifier_agent,
        app_name="refinement_verifier",
    )

    refiner_runner = InMemoryRunner(
        agent=refiner,
        app_name="refinement_refiner",
    )

    user_id = "test_user"
    iterations = 0
    feedback = None

    while True:
        # Verify the current explanation
        verifier_session = await verifier_runner.session_service.create_session(
            app_name="refinement_verifier",
            user_id=user_id,
            state={
                "original_text": original_text,
                "simplified_text": simplified_text,
            },
        )

        verifier_input = json.dumps({
            "original_text": original_text,
            "simplified_text": simplified_text,
            "clinical_atoms": clinical_atoms if clinical_atoms is not None else [],
            "target_grade_level": target_grade_level,
        })

        async for _ in verifier_runner.run_async(
            user_id=user_id,
            session_id=verifier_session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=verifier_input)],
            ),
        ):
            pass

        verifier_session = await verifier_runner.session_service.get_session(
            app_name="refinement_verifier",
            user_id=user_id,
            session_id=verifier_session.id,
        )

        feedback = verifier_session.state["verifier_feedback"]

        # Stop if verification passes
        if feedback["overall_pass"]:
            break

        # Stop if we've already used both refinement attempts
        if iterations >= MAX_ITERATIONS:
            break

        # Refine using the Verifier's feedback
        refiner_session = await refiner_runner.session_service.create_session(
            app_name="refinement_refiner",
            user_id=user_id,
            state={
                "original_text": original_text,
                "simplified_text": simplified_text,
                "verifier_feedback": feedback,
            },
        )

        async for _ in refiner_runner.run_async(
            user_id=user_id,
            session_id=refiner_session.id,
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text="Refine the simplified medical explanation."
                    )
                ],
            ),
        ):
            pass

        refiner_session = await refiner_runner.session_service.get_session(
            app_name="refinement_refiner",
            user_id=user_id,
            session_id=refiner_session.id,
        )

        simplified_text = refiner_session.state["simplified_text"]
        iterations += 1

    return {
        "simplified_text": simplified_text,
        "verifier_feedback": feedback,
        "overall_pass": feedback["overall_pass"],
        "iterations": iterations,
    }