import unittest

from agents.refinement_loop import run_refinement_loop


def feedback(passed):
    return {
        "faithfulness": {"passed": passed, "issues": []},
        "omissions": {"passed": True, "issues": []},
        "unsupported_additions": {"passed": True, "issues": []},
        "reading_level": {"passed": True, "estimated_grade": 7, "issues": []},
        "overall_pass": passed,
        "refiner_instructions": [],
    }


class VerifierRefinerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_refiner_revision_is_reverified(self):
        calls = []
        verifier_calls = 0

        async def fake_executor(agent, state, prompt):
            nonlocal verifier_calls
            calls.append(agent.name)
            if agent.name == "verifier_agent":
                verifier_calls += 1
                return {"verifier_feedback": feedback(verifier_calls > 1)}
            if agent.name == "Refiner":
                self.assertEqual(state["simplified_text"], "Wrong dose.")
                return {"simplified_text": "Correct dose."}
            self.fail(f"Unexpected agent called: {agent.name}")

        result = await run_refinement_loop(
            "The correct dose is 10 mg.",
            "Wrong dose.",
            agent_executor=fake_executor,
        )

        self.assertEqual(result["simplified_text"], "Correct dose.")
        self.assertTrue(result["overall_pass"])
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(
            calls,
            ["verifier_agent", "Refiner", "verifier_agent"],
        )


if __name__ == "__main__":
    unittest.main()
