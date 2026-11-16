"""Synthetic adversarial examples for live Verifier calibration."""

from typing import TypedDict


class VerifierCase(TypedDict):
    name: str
    category: str
    source_text: str
    candidate_text: str


ADVERSARIAL_CASES: list[VerifierCase] = [
    {
        "name": "reported status changed to direct assertion",
        "category": "faithfulness",
        "source_text": "The patient reports no fever.",
        "candidate_text": "The patient has no fever.",
    },
    {
        "name": "suspected diagnosis changed to confirmed",
        "category": "faithfulness",
        "source_text": "A viral infection is suspected.",
        "candidate_text": "A viral infection is confirmed.",
    },
    {
        "name": "negation reversed",
        "category": "faithfulness",
        "source_text": "The patient denies chest pain.",
        "candidate_text": "The patient has chest pain.",
    },
    {
        "name": "medication dose changed",
        "category": "faithfulness",
        "source_text": "Take 5 mg once daily.",
        "candidate_text": "Take 50 mg once daily.",
    },
    {
        "name": "symptom duration changed",
        "category": "faithfulness",
        "source_text": "The cough has lasted for three days.",
        "candidate_text": "The cough has lasted for three weeks.",
    },
    {
        "name": "important plan omitted",
        "category": "omissions",
        "source_text": "The plan is rest, fluids, and follow-up if symptoms worsen.",
        "candidate_text": "The plan is rest.",
    },
    {
        "name": "unsupported diagnosis added",
        "category": "unsupported_additions",
        "source_text": "The patient reports a dry cough.",
        "candidate_text": "The patient reports a dry cough and has pneumonia.",
    },
]
