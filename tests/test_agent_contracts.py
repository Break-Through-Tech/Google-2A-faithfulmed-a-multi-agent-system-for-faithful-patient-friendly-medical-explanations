import unittest

from agents.contracts import PipelineCase
from agents.verifier import VerifierInput


class PipelineCaseTests(unittest.TestCase):
    def setUp(self):
        self.source_text = "Patient reports cough. No fever is reported."
        self.extractor_output = {
            "atoms": [
                {
                    "text": "Patient reports cough",
                    "type": "symptom",
                    "source_quote": "Patient reports cough",
                },
                {
                    "text": "No fever is reported",
                    "type": "negative_finding",
                    "source_quote": "No fever is reported",
                },
            ],
            "not_found_types": [],
        }

    def test_builds_canonical_case_and_adk_state(self):
        case = PipelineCase.from_extractor_output(
            self.source_text,
            self.extractor_output,
        )

        self.assertEqual(case.atoms[0].id, "atom-001")
        self.assertEqual(case.to_adk_state()["target_grade"], 8)
        self.assertEqual(case.verifier_atoms()[0]["concept"], "Patient reports cough")

    def test_rejects_source_quote_not_in_source(self):
        self.extractor_output["atoms"][0]["source_quote"] = "Patient has pneumonia"

        with self.assertRaisesRegex(ValueError, "source_quote"):
            PipelineCase.from_extractor_output(
                self.source_text,
                self.extractor_output,
            )

    def test_maps_candidate_to_refiner_contract(self):
        case = PipelineCase.from_extractor_output(
            self.source_text,
            self.extractor_output,
        )

        state = case.refiner_state("A patient-friendly draft.", {"overall_pass": False})

        self.assertEqual(state["original_text"], self.source_text)
        self.assertEqual(state["simplified_text"], "A patient-friendly draft.")

    def test_verifier_adapter_matches_existing_input_schema(self):
        case = PipelineCase.from_extractor_output(
            self.source_text,
            self.extractor_output,
        )

        verifier_input = VerifierInput(
            original_text=case.source_text,
            simplified_text="The patient reports a cough. No fever is reported.",
            clinical_atoms=case.verifier_atoms(),
        )

        self.assertEqual(verifier_input.clinical_atoms[0].concept, "Patient reports cough")


if __name__ == "__main__":
    unittest.main()
