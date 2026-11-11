import unittest

from agents.model_runner import MODEL
from agents.refiner import refiner


class RefinerAgentTests(unittest.TestCase):
    def test_refiner_has_direct_model_call_contract(self):
        self.assertEqual(refiner.name, "Refiner")
        self.assertEqual(refiner.output_key, "simplified_text")
        self.assertEqual(refiner.model, MODEL)
        self.assertIn("verifier_feedback", refiner.instruction)
        self.assertIsNone(refiner.output_schema)


if __name__ == "__main__":
    unittest.main()
