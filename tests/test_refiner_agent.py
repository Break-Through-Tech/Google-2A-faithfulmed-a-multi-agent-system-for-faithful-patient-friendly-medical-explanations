import asyncio
import json

from dotenv import load_dotenv

load_dotenv()


from google.adk.runners import InMemoryRunner
from google.genai import types
from agents.refiner import refiner

async def main():
    runner = InMemoryRunner(
        agent=refiner,
        app_name="refiner_test"
    )

    session = await runner.session_service.create_session(
    app_name="refiner_test",
    user_id="test_user",
    state={
    "original_text": (
        "The patient has hypertension and was prescribed "
        "lisinopril 10 mg daily."
    ),
    "simplified_text": (
        "The patient has high blood pressure and was prescribed "
        "lisinopril 20 mg daily."
    ),
    "verifier_feedback": json.dumps({
        "faithfulness": {
            "passed": False,
            "issues": [
                {
                    "description": "Incorrect lisinopril dosage",
                    "severity": "high",
                    "source_evidence": "lisinopril 10 mg daily",
                    "simplified_evidence": "lisinopril 20 mg daily",
                    "atom_id": None
                }
            ]
        },
        "omissions": {
            "passed": True,
            "issues": []
        },
        "unsupported_additions": {
            "passed": True,
            "issues": []
        },
        "reading_level": {
            "passed": True,
            "estimated_grade": 7,
            "issues": []
        },
        "overall_pass": False,
        "refiner_instructions": [
            "Correct lisinopril dosage from 20 mg to 10 mg."
        ]
    })
}
)
    message = types.Content(
    role="user",
    parts=[
        types.Part(text="Refine the simplified explanation.")
    ]
)

    async for event in runner.run_async(
    user_id="test_user",
    session_id=session.id,
    new_message=message,
):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    print("REFINER OUTPUT:")
                    print(part.text)


if __name__ == "__main__":
    asyncio.run(main())