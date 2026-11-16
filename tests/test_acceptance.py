import unittest

from agents.acceptance import assess_acceptance
from agents.adversarial_cases import ADVERSARIAL_CASES


PASSING_VERDICT = {
    "overall_pass": True,
    "faithfulness": {"passed": True, "issues": []},
    "omissions": {"passed": True, "issues": []},
    "unsupported_additions": {"passed": True, "issues": []},
    "reading_level": {"passed": True, "issues": []},
}


class AcceptanceTests(unittest.TestCase):
    def test_requires_human_review_even_when_automated_signals_pass(self):
        result = assess_acceptance(
            "The patient reports a cough for three days.",
            "The patient reports a cough for 3 days.",
            PASSING_VERDICT,
        )

        self.assertTrue(result["deterministic_checks_passed"])
        self.assertTrue(result["verifier_passed"])
        self.assertEqual(result["review_status"], "human_review_required")

    def test_flags_reported_finding_rewritten_as_established(self):
        case = ADVERSARIAL_CASES[0]
        result = assess_acceptance(
            case["source_text"],
            case["candidate_text"],
            PASSING_VERDICT,
        )

        codes = {issue["code"] for issue in result["deterministic_issues"]}
        self.assertIn("reported_status_not_preserved", codes)

    def test_flags_confirmed_wording_when_source_is_uncertain(self):
        case = ADVERSARIAL_CASES[1]
        result = assess_acceptance(
            case["source_text"],
            case["candidate_text"],
            PASSING_VERDICT,
        )

        codes = {issue["code"] for issue in result["deterministic_issues"]}
        self.assertIn("uncertainty_not_preserved", codes)
        self.assertIn("certainty_strengthened", codes)

    def test_flags_negation_reversal(self):
        case = ADVERSARIAL_CASES[2]
        result = assess_acceptance(
            case["source_text"],
            case["candidate_text"],
            PASSING_VERDICT,
        )

        codes = {issue["code"] for issue in result["deterministic_issues"]}
        self.assertIn("negation_not_preserved", codes)

    def test_flags_changed_dose_and_timing(self):
        for index in (3, 4):
            with self.subTest(case=ADVERSARIAL_CASES[index]["name"]):
                case = ADVERSARIAL_CASES[index]
                result = assess_acceptance(
                    case["source_text"],
                    case["candidate_text"],
                    PASSING_VERDICT,
                )
                codes = {issue["code"] for issue in result["deterministic_issues"]}
                self.assertIn("quantity_or_timing_changed", codes)

    def test_flags_missing_user_marked_critical_fact(self):
        case = ADVERSARIAL_CASES[5]
        result = assess_acceptance(
            case["source_text"],
            case["candidate_text"],
            PASSING_VERDICT,
            protected_facts=["fluids", "follow-up if symptoms worsen"],
        )

        self.assertEqual(
            sum(issue["code"] == "protected_fact_missing"
                for issue in result["deterministic_issues"]),
            2,
        )

    def test_surfaces_verifier_addition_issue(self):
        case = ADVERSARIAL_CASES[6]
        verdict = {
            **PASSING_VERDICT,
            "overall_pass": False,
            "unsupported_additions": {
                "passed": False,
                "issues": [{
                    "description": "Pneumonia is not supported by the source.",
                    "severity": "high",
                }],
            },
        }
        result = assess_acceptance(
            case["source_text"],
            case["candidate_text"],
            verdict,
        )

        self.assertFalse(result["verifier_passed"])
        self.assertEqual(result["verifier_issues"][0]["category"], "unsupported_additions")
        self.assertEqual(result["review_status"], "human_review_required")


if __name__ == "__main__":
    unittest.main()
