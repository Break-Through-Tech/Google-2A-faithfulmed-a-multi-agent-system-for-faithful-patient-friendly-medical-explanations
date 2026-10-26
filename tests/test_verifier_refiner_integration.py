
import asyncio
import json

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.verifier import verifier_agent
from agents.refiner import refiner

load_dotenv()


async def main():
    # 1. Create intentionally incorrect medical text
    state = {
        "original_text": (
            "The patient has hypertension and was prescribed "
            "lisinopril 10 mg daily."
        ),
        "simplified_text": (
            "The patient has high blood pressure and was prescribed "
            "lisinopril 20 mg daily."
        ),
    }

    # 2. Create the Verifier runner
    verifier_runner = InMemoryRunner(
        agent=verifier_agent,
        app_name="integration_test"
    )

    verifier_session = await verifier_runner.session_service.create_session(
        app_name="integration_test",
        user_id="test_user",
        state=state,
    )

    # 3. Run the Verifier
    print("\n--- VERIFIER ---")

    verifier_input = json.dumps({
        "original_text": state["original_text"],
        "simplified_text": state["simplified_text"],
        "clinical_atoms": [],
        "target_grade_level": 8,
    })

    async for event in verifier_runner.run_async(
        user_id="test_user",
        session_id=verifier_session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=verifier_input)],
        ),
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts or []:
                if part.text:
                    print(part.text)

    # 4. Retrieve the Verifier's feedback
    verifier_session = await verifier_runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=verifier_session.id,
    )

    feedback = verifier_session.state.get("verifier_feedback")

    print("\n--- STORED FEEDBACK ---")
    print("Type:", type(feedback).__name__)
    print(json.dumps(feedback, indent=2, default=str))

    if not isinstance(feedback, dict):
        raise ValueError("Verifier did not store structured feedback.")

    if feedback.get("overall_pass") is True:
        print("\nVerification passed. Refinement is unnecessary.")
        return

    # 5. Create the Refiner runner
    print("\n--- REFINER ---")

    refiner_runner = InMemoryRunner(
        agent=refiner,
        app_name="refiner_integration_test"
    )

    # 6. Copy the Verifier's state into the Refiner session
    refiner_session = await refiner_runner.session_service.create_session(
        app_name="refiner_integration_test",
        user_id="test_user",
        state=verifier_session.state.copy(),
    )

    # 7. Run the Refiner
    async for event in refiner_runner.run_async(
        user_id="test_user",
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
        if event.is_final_response() and event.content:
            for part in event.content.parts or []:
                if part.text:
                    print(part.text)

    # 8. Retrieve the updated simplified text
    refiner_session = await refiner_runner.session_service.get_session(
        app_name="refiner_integration_test",
        user_id="test_user",
        session_id=refiner_session.id,
    )

    print("\n--- UPDATED SIMPLIFIED TEXT ---")
    print(refiner_session.state.get("simplified_text"))

        # Get the corrected explanation from the Refiner
    refined_text = refiner_session.state["simplified_text"]

    # Create a new Verifier session using the corrected text
    recheck_session = await verifier_runner.session_service.create_session(
        app_name="integration_test",
        user_id="test_user",
        state={
            "original_text": state["original_text"],
            "simplified_text": refined_text,
        },
    )

    # Prepare the input expected by the Verifier
    recheck_input = json.dumps({
        "original_text": state["original_text"],
        "simplified_text": refined_text,
        "clinical_atoms": [],
        "target_grade_level": 8,
    })

    print("\n--- RE-VERIFICATION ---")

    # Run the Verifier again on the corrected explanation
    async for event in verifier_runner.run_async(
        user_id="test_user",
        session_id=recheck_session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=recheck_input)],
        ),
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts or []:
                if part.text:
                    print(part.text)

    # Retrieve the Verifier's final feedback
    recheck_session = await verifier_runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=recheck_session.id,
    )

    final_feedback = recheck_session.state["verifier_feedback"]

    print("\n--- FINAL RESULT ---")
    print("Overall pass:", final_feedback["overall_pass"])

    assert final_feedback["overall_pass"] is True, (
        "The refined explanation did not pass re-verification."
    )


if __name__ == "__main__":
    asyncio.run(main())
