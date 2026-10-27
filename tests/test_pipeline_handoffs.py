import json
import unittest

from agents.run_pipeline import build_synthetic_case, run_pipeline


def verifier_feedback(passed, issue=None):
    issue = {
        "description": "A planted fact changed." if issue is None else issue["description"],
        "severity": "high",
        "source_evidence": "No fever" if issue is None else issue["source_evidence"],
        "simplified_evidence": (
            "The patient has a fever" if issue is None else issue["simplified_evidence"]
        ),
        "atom_id": "atom-002" if issue is None else issue["atom_id"],
    }
    return {
        "faithfulness": {
            "passed": passed,
            "issues": [] if passed else [issue],
        },
        "omissions": {"passed": True, "issues": []},
        "unsupported_additions": {"passed": True, "issues": []},
        "reading_level": {
            "passed": True,
            "estimated_grade": 7.0,
            "issues": [],
        },
        "overall_pass": passed,
        "refiner_instructions": [] if passed else ["Restore the negated fever finding."],
    }


class PipelineHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_flags_require_review_without_triggering_refiner(self):
        case = build_synthetic_case()
        calls = []

        async def fake_executor(agent, app_name, state, prompt):
            calls.append(agent.name)
            if agent.name == "simplifier_agent":
                return {"draft": "A patient-friendly draft."}
            if agent.name == "readability_agent":
                return {"readability_draft": "A patient-friendly candidate."}
            if agent.name == "verifier_agent":
                return {"verifier_feedback": verifier_feedback(True)}
            self.fail(f"Unexpected agent called: {agent.name}")

        result = await run_pipeline(case, agent_executor=fake_executor)

        self.assertFalse(result["acceptance"]["deterministic_checks_passed"])
        self.assertEqual(result["refinements"], 0)
        self.assertEqual(result["final_text"], "A patient-friendly candidate.")
        self.assertEqual(
            calls,
            ["simplifier_agent", "readability_agent", "verifier_agent"],
        )

    async def test_passes_case_between_agents_and_reverifies_refinement(self):
        case = build_synthetic_case()
        calls = []
        verifier_calls = 0

        async def fake_executor(agent, app_name, state, prompt):
            nonlocal verifier_calls
            calls.append((agent.name, state, prompt))
            if agent.name == "simplifier_agent":
                self.assertEqual(state["source_text"], case.source_text)
                self.assertEqual(len(state["atoms"]), 6)
                self.assertEqual(state["retrieved_context"], [])
                return {
                    "draft": (
                        "The patient reports a dry cough for three days. No fever or "
                        "shortness of breath is reported. A viral illness is suspected."
                    )
                }
            if agent.name == "readability_agent":
                self.assertIn("draft", state)
                self.assertEqual(state["target_grade"], 8)
                self.assertEqual(state["protected_facts"], [])
                return {
                    "readability_draft": (
                        "The patient reports a dry cough for three days. No fever or "
                        "shortness of breath is reported. A viral illness is suspected."
                    )
                }
            if agent.name == "verifier_agent":
                verifier_calls += 1
                verifier_input = json.loads(prompt)
                self.assertEqual(
                    verifier_input["clinical_atoms"][0]["id"],
                    "atom-001",
                )
                self.assertEqual(verifier_input["original_text"], case.source_text)
                return {
                    "verifier_feedback": verifier_feedback(verifier_calls > 1)
                }
            if agent.name == "Refiner":
                self.assertEqual(state["original_text"], case.source_text)
                self.assertEqual(
                    state["simplified_text"],
                    "The patient reports a dry cough for three days. No fever or "
                    "shortness of breath is reported. A viral illness is suspected.",
                )
                self.assertFalse(state["verifier_feedback"]["overall_pass"])
                self.assertNotIn(
                    "deterministic_review",
                    state["verifier_feedback"],
                )
                return {
                    "simplified_text": (
                        "The patient reports a dry cough for three days. No fever or "
                        "shortness of breath is reported. A viral infection is suspected."
                    )
                }
            self.fail(f"Unexpected agent called: {agent.name}")

        result = await run_pipeline(case, agent_executor=fake_executor)

        self.assertEqual(result["refinements"], 1)
        self.assertTrue(result["overall_pass"])
        self.assertEqual(verifier_calls, 2)
        self.assertEqual(
            [call[0] for call in calls],
            [
                "simplifier_agent",
                "readability_agent",
                "verifier_agent",
                "Refiner",
                "verifier_agent",
            ],
        )

    async def test_stops_after_configured_maximum_refinements(self):
        calls = []

        async def fake_executor(agent, app_name, state, prompt):
            calls.append(agent.name)
            if agent.name == "simplifier_agent":
                return {"draft": "Synthetic draft."}
            if agent.name == "readability_agent":
                return {"readability_draft": "Synthetic readable draft."}
            if agent.name == "verifier_agent":
                return {"verifier_feedback": verifier_feedback(False)}
            if agent.name == "Refiner":
                return {"simplified_text": "Synthetic refined draft."}
            self.fail(f"Unexpected agent called: {agent.name}")

        result = await run_pipeline(
            build_synthetic_case(),
            max_refinements=2,
            agent_executor=fake_executor,
        )

        self.assertFalse(result["overall_pass"])
        self.assertEqual(result["refinements"], 2)
        self.assertEqual(calls.count("Refiner"), 2)
        self.assertEqual(calls.count("verifier_agent"), 3)

    async def test_exercise_refiner_injects_error_and_tracks_reverification(self):
        verifier_calls = 0
        refiner_input = {}

        async def fake_executor(agent, app_name, state, prompt):
            nonlocal verifier_calls, refiner_input
            if agent.name == "simplifier_agent":
                return {
                    "draft": (
                        "The patient reports a dry cough for three days. No fever or "
                        "shortness of breath is reported. A viral infection is suspected."
                    )
                }
            if agent.name == "readability_agent":
                return {
                    "readability_draft": (
                        "The patient reports a dry cough for three days. No fever or "
                        "shortness of breath is reported. A viral infection is suspected."
                    )
                }
            if agent.name == "verifier_agent":
                verifier_calls += 1
                verifier_input = json.loads(prompt)
                if verifier_calls == 1:
                    self.assertIn(
                        "The suspected viral infection is confirmed.",
                        verifier_input["simplified_text"],
                    )
                    return {
                        "verifier_feedback": verifier_feedback(
                            False,
                            {
                                "description": "Suspected diagnosis was changed to confirmed.",
                                "source_evidence": "is suspected",
                                "simplified_evidence": "is confirmed",
                                "atom_id": "atom-004",
                            },
                        )
                    }
                self.assertNotIn(
                    "The suspected viral infection is confirmed.",
                    verifier_input["simplified_text"],
                )
                return {"verifier_feedback": verifier_feedback(True)}
            if agent.name == "Refiner":
                refiner_input = state
                self.assertIn(
                    "The suspected viral infection is confirmed.",
                    state["simplified_text"],
                )
                return {
                    "simplified_text": (
                        "The patient reports a dry cough for three days. No fever or "
                        "shortness of breath is reported. A viral infection is suspected."
                    )
                }
            self.fail(f"Unexpected agent called: {agent.name}")

        result = await run_pipeline(
            build_synthetic_case(),
            exercise_refiner=True,
            agent_executor=fake_executor,
        )

        self.assertTrue(result["refiner_demo"])
        self.assertTrue(result["refiner_exercised"])
        self.assertEqual(result["refinements"], 1)
        self.assertFalse(result["initial_verifier_feedback"]["overall_pass"])
        self.assertTrue(result["overall_pass"])
        self.assertEqual(verifier_calls, 2)
        self.assertIn(
            "The suspected viral infection is confirmed.",
            result["initial_candidate"],
        )
        self.assertNotIn(
            "The suspected viral infection is confirmed.",
            result["final_text"],
        )
        self.assertEqual(
            refiner_input["verifier_feedback"]["overall_pass"],
            False,
        )

    async def test_rejects_negative_refinement_limit(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            await run_pipeline(build_synthetic_case(), max_refinements=-1)


if __name__ == "__main__":
    unittest.main()
