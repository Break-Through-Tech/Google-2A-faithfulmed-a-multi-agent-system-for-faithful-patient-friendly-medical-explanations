import os
import unittest

from agents.readability.agent import root_agent
from agents.readability.metrics import assess_candidate, measure_readability


class ReadabilityMetricsTests(unittest.TestCase):
    def test_reports_multiple_readability_measures(self):
        result = measure_readability(
            "Take the pill with water. Call your doctor if your fever gets worse."
        )

        self.assertEqual(result.sentence_count, 2)
        self.assertGreater(result.word_count, 0)
        self.assertGreaterEqual(result.difficult_word_ratio, 0)
        self.assertLessEqual(result.difficult_word_ratio, 1)
        self.assertGreaterEqual(result.flesch_reading_ease, 0)
        self.assertIsInstance(result.smog_index, float)
        self.assertIsInstance(result.coleman_liau_index, float)
        self.assertIsInstance(result.automated_readability_index, float)
        self.assertIsInstance(result.gunning_fog_index, float)
        self.assertIsInstance(result.dale_chall_score, float)

    def test_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            measure_readability("  ")

    def test_assessment_keeps_rewrite_when_protected_facts_are_preserved(self):
        original = "Take metformin 500 mg twice daily."
        candidate = "Take metformin 500 mg twice a day."

        result = assess_candidate(
            original,
            candidate,
            protected_facts=["metformin", "500 mg"],
        )

        self.assertEqual(result.output_text, candidate)
        self.assertTrue(result.preservation_passed)
        self.assertEqual(result.status, "accepted")

    def test_assessment_falls_back_if_a_protected_fact_is_lost(self):
        original = "Take metformin 500 mg twice daily."
        candidate = "Take the medicine twice daily."

        result = assess_candidate(
            original,
            candidate,
            protected_facts=["metformin", "500 mg"],
        )

        self.assertEqual(result.output_text, original)
        self.assertFalse(result.preservation_passed)
        self.assertEqual(result.status, "preservation_failed")
        self.assertEqual(result.missing_protected_facts, ("metformin", "500 mg"))

    def test_assessment_logs_metrics_without_logging_clinical_text(self):
        clinical_text = "Private sample phrase: take metformin 500 mg."

        with self.assertLogs("faithfulmed.readability", level="INFO") as captured:
            assess_candidate(clinical_text, clinical_text, ["metformin", "500 mg"])

        self.assertNotIn(clinical_text, "\n".join(captured.output))
        self.assertIn("flesch_kincaid_grade", "\n".join(captured.output))

    def test_grade_target_is_reported_without_claiming_success(self):
        text = (
            "The patient demonstrated significant improvement in cardiovascular "
            "symptomatology following administration of the prescribed medication."
        )

        result = assess_candidate(text, text, target_grade=1)

        self.assertFalse(result.meets_grade_target)
        self.assertEqual(result.status, "target_not_met")


class ReadabilityAgentTests(unittest.TestCase):
    def test_adk_agent_has_expected_contract(self):
        self.assertEqual(root_agent.name, "readability_agent")
        self.assertEqual(root_agent.output_key, "readability_draft")
        self.assertEqual(
            root_agent.model,
            os.getenv("FAITHFULMED_MODEL", "gemini-flash-latest"),
        )
        self.assertIsInstance(root_agent.instruction, str)
        self.assertIn("{draft?}", root_agent.instruction)


if __name__ == "__main__":
    unittest.main()
