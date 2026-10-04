"""
FaithfulMed — starter evaluation harness.

Build this out in Week 1-2, BEFORE the agents. You can't improve what you can't measure.
Run: python notebooks/eval_harness.py  (expects data/medaesqa_v1.json — see data/README.md)

Provides:
  - readability_scores(text): Flesch-Kincaid grade, SMOG, jargon density, length
  - meets_readability_target(text): the <= 8th-grade success criterion
  - load_medaesqa(path): loads the gold eval set
  - verifier_agreement(human, verifier): accuracy + Cohen's kappa for Verifier calibration

This is model-agnostic — it scores text, whatever produced it. Extend, don't treat as final.
Each agent owner plugs their agent's outputs into this harness.
"""
from __future__ import annotations
import json
from pathlib import Path

import textstat
from sklearn.metrics import cohen_kappa_score, accuracy_score

DATA = Path(__file__).resolve().parent.parent / "data" / "medaesqa_v1.json"


def readability_scores(text: str) -> dict:
    """Core readability metrics. Jargon density ~ fraction of 'difficult' (non-familiar) words.

    difficult_words() uses the Dale-Chall familiar-word list as a proxy for jargon; swap in a
    UMLS/medical-term lookup later for a clinically grounded jargon score.
    """
    words = max(textstat.lexicon_count(text, removepunct=True), 1)
    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "jargon_density": textstat.difficult_words(text) / words,
        "word_count": words,
    }


def meets_readability_target(text: str, max_grade: float = 8.0) -> bool:
    """Success criterion: <= 8th-grade reading level (Flesch-Kincaid)."""
    return textstat.flesch_kincaid_grade(text) <= max_grade

def is_refusal(text: str) -> bool:
    """Return True if the model appears to refuse the requested task."""
    refusal_phrases = [
        "i can't provide",
        "i cannot provide",
        "i can't assist",
        "i cannot assist",
        "i can't help",
        "i cannot help",
        "i'm unable to",
        "i am unable to",
        "i'm not able to",
        "i am not able to",
        "i can't answer",
        "i cannot answer",
    ]

    text_lower = text.lower().strip()

    return any(phrase in text_lower for phrase in refusal_phrases)

def evaluate_outputs(outputs: list[str]) -> dict:
    """Evaluate model outputs and return aggregate metrics."""

    if not outputs:
        raise ValueError("outputs cannot be empty")

    # Separate refusals from valid responses
    valid_outputs = [text for text in outputs if not is_refusal(text)]
    num_refusals = len(outputs) - len(valid_outputs)

    # Calculate refusal rate using ALL outputs
    refusal_rate = num_refusals / len(outputs)

    # Handle the case where every output was a refusal
    if not valid_outputs:
        return {
            "num_outputs": len(outputs),
            "num_valid_outputs": 0,
            "num_refusals": num_refusals,
            "refusal_rate": refusal_rate,
        }

    # Calculate readability only on valid responses
    scores = [readability_scores(text) for text in valid_outputs]

    return {
        "num_outputs": len(outputs),
        "num_valid_outputs": len(valid_outputs),
        "num_refusals": num_refusals,
        "refusal_rate": refusal_rate,

        "avg_flesch_kincaid": sum(
            score["flesch_kincaid_grade"] for score in scores
        ) / len(scores),

        "percent_at_or_below_grade_8": sum(
            meets_readability_target(text) for text in valid_outputs
        ) / len(valid_outputs) * 100,

        "avg_smog": sum(
            score["smog_index"] for score in scores
        ) / len(scores),

        "avg_jargon_density": sum(
            score["jargon_density"] for score in scores
        ) / len(scores),

        "avg_word_count": sum(
            score["word_count"] for score in scores
        ) / len(scores),
    }

def load_medaesqa(path: Path = DATA) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download medaesqa_v1.json from https://osf.io/ydbzq "
            f"into the data/ folder (see data/README.md)."
        )
    with open(path) as f:
        return json.load(f)


def verifier_agreement(human_labels: list[int], verifier_labels: list[int]) -> dict:
    """Calibrate the Verifier against human faithfulness labels.

    Targets from the success criteria: accuracy >= 0.80, Cohen's kappa >= 0.6.
    Pass 0/1 (or categorical) labels of equal length.
    """
    return {
        "accuracy": accuracy_score(human_labels, verifier_labels),
        "cohen_kappa": cohen_kappa_score(human_labels, verifier_labels),
        "n": len(human_labels),
    }


if __name__ == "__main__":
    demo = ("Your discharge summary says you were prescribed a beta-blocker to manage "
            "hypertension and should follow up with cardiology in two weeks.")
    print("Readability demo:", readability_scores(demo))
    print("Meets <=8th grade:", meets_readability_target(demo))

    # Tiny agreement demo (replace with real Verifier vs. human labels):
    print("Agreement demo:", verifier_agreement([1, 1, 0, 1, 0], [1, 0, 0, 1, 0]))

    try:
        data = load_medaesqa()
        print(f"Loaded MedAESQA: {len(data)} questions.")
    except FileNotFoundError as e:
        print(e)