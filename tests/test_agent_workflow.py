import json
import unittest

from agents.extractor_agent import (
    ALLOWED_TYPES,
    validate_extractor_output,
)
from agents.step_runner import run_agent_stage


SOURCE = "The patient reports a cough for three days. No fever is reported."
EXTRACTION = {
    "atoms": [
        {
            "text": "The patient reports a cough for three days",
            "type": "symptom",
            "source_quote": "reports a cough for three days",
        },
        {
            "text": "No fever is reported",
            "type": "negative_finding",
            "source_quote": "No fever is reported",
        },
    ],
    "not_found_types": sorted(ALLOWED_TYPES - {"symptom", "negative_finding"}),
}


def verifier_feedback(passed):
    return {
        "faithfulness": {"passed": passed, "issues": []},
        "omissions": {"passed": True, "issues": []},
        "unsupported_additions": {"passed": True, "issues": []},
        "reading_level": {"passed": True, "estimated_grade": 7, "issues": []},
        "overall_pass": passed,
        "refiner_instructions": [] if passed else ["Restore the reported cough duration."],
    }


class ExtractorOutputTests(unittest.TestCase):
    def test_accepts_source_quoted_facts_and_complete_missing_categories(self):
        result = validate_extractor_output(EXTRACTION, SOURCE)

        self.assertEqual(len(result.atoms), 2)
        self.assertEqual(result.atoms[0].source_quote, "reports a cough for three days")

    def test_rejects_an_extractor_quote_not_in_the_source(self):
        output = {
            "atoms": [
                {
                    "text": "The patient has pneumonia",
                    "type": "diagnosis",
                    "source_quote": "has pneumonia",
                }
            ],
            "not_found_types": sorted(ALLOWED_TYPES - {"diagnosis"}),
        }

        with self.assertRaisesRegex(ValueError, "source_quote"):
            validate_extractor_output(output, SOURCE)


class StepRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_agents_individually_and_hands_each_output_forward(self):
        calls = []

        async def fake_executor(agent, state, prompt):
            calls.append((agent.name, state, prompt))
            if agent.name == "extractor_agent":
                self.assertEqual(state["source_text"], SOURCE)
                return {"extractor_output": EXTRACTION}
            if agent.name == "simplifier_agent":
                self.assertEqual(state["source_text"], SOURCE)
                return {"draft": "The patient reports a cough for three days. No fever."}
            if agent.name == "readability_agent":
                self.assertEqual(
                    state["draft"],
                    "The patient reports a cough for three days. No fever.",
                )
                return {
                    "readability_draft": (
                        "The patient reports a cough for three days. No fever."
                    )
                }
            if agent.name == "verifier_agent":
                self.assertEqual(
                    json.loads(prompt)["simplified_text"],
                    "The patient reports a cough for three days. No fever.",
                )
                return {"verifier_feedback": verifier_feedback(True)}
            self.fail(f"Unexpected agent called: {agent.name}")

        state = {}
        expected_next = ["Simplifier", "Readability", "Verifier", None]
        for stage, next_stage in zip(
            ("Extractor", "Simplifier", "Readability", "Verifier"),
            expected_next,
        ):
            state, result = await run_agent_stage(
                stage,
                SOURCE,
                [],
                8,
                state,
                agent_executor=fake_executor,
            )
            self.assertEqual(result["next_agent"], next_stage)
            self.assertIn("input", result)
            self.assertIn("output", result)

        self.assertEqual(
            [call[0] for call in calls],
            [
                "extractor_agent",
                "simplifier_agent",
                "readability_agent",
                "verifier_agent",
            ],
        )
        self.assertEqual(len(state["history"]), 4)

    async def test_refiner_output_is_fed_back_to_verifier(self):
        verifier_calls = 0

        async def fake_executor(agent, state, _prompt):
            nonlocal verifier_calls
            if agent.name == "extractor_agent":
                return {"extractor_output": EXTRACTION}
            if agent.name == "simplifier_agent":
                return {"draft": "The patient reports a cough for three days."}
            if agent.name == "readability_agent":
                return {"readability_draft": "The patient has a cough."}
            if agent.name == "verifier_agent":
                verifier_calls += 1
                self.assertEqual(
                    state["simplified_text"],
                    "The patient has a cough."
                    if verifier_calls == 1
                    else "The patient reports a cough for three days.",
                )
                return {
                    "verifier_feedback": verifier_feedback(verifier_calls > 1)
                }
            if agent.name == "Refiner":
                self.assertEqual(state["simplified_text"], "The patient has a cough.")
                return {
                    "simplified_text": (
                        "The patient reports a cough for three days."
                    )
                }
            self.fail(f"Unexpected agent called: {agent.name}")

        state = {}
        for stage in ("Extractor", "Simplifier", "Readability", "Verifier"):
            state, _ = await run_agent_stage(
                stage,
                SOURCE,
                [],
                8,
                state,
                agent_executor=fake_executor,
            )
        self.assertEqual(state["next_stage"], "Refiner")

        state, refiner_result = await run_agent_stage(
            "Refiner",
            SOURCE,
            [],
            8,
            state,
            agent_executor=fake_executor,
        )
        self.assertEqual(refiner_result["next_agent"], "Verifier")
        self.assertEqual(state["candidate"], "The patient reports a cough for three days.")

        state, _ = await run_agent_stage(
            "Verifier",
            SOURCE,
            [],
            8,
            state,
            agent_executor=fake_executor,
        )
        self.assertIsNone(state["next_stage"])
        self.assertEqual(verifier_calls, 2)

    async def test_rejects_running_a_later_agent_before_its_prerequisite(self):
        with self.assertRaisesRegex(ValueError, "Run the Extractor"):
            await run_agent_stage("Simplifier", SOURCE, [], 8, {})


if __name__ == "__main__":
    unittest.main()
