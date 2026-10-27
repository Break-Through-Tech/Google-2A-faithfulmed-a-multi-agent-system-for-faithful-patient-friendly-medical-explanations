import unittest

from agents.simplifier.agent import root_agent


class SimplifierAgentTests(unittest.TestCase):
    def test_uses_shared_draft_output_key(self):
        self.assertEqual(root_agent.output_key, "draft")

    def test_treats_retrieval_as_optional_general_context(self):
        instruction = root_agent.instruction

        self.assertIn("{source_text?}", instruction)
        self.assertIn("{atoms?}", instruction)
        self.assertIn("{retrieved_context?}", instruction)
        self.assertIn("If retrieved_context is absent or empty", instruction)


if __name__ == "__main__":
    unittest.main()
