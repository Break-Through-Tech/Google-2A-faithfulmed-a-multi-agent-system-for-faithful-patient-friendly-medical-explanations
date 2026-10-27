"""Evaluate a CSV column of generated patient-facing text."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from agents.readability.metrics import measure_readability


def evaluate_csv(path: Path, text_column: str) -> dict[str, int | float]:
    """Return aggregate readability metrics for non-empty rows in a CSV."""
    with path.open(newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain a {text_column!r} column")
        texts = [
            row[text_column].strip()
            for row in reader
            if row.get(text_column) and row[text_column].strip()
        ]

    if not texts:
        raise ValueError(f"CSV contains no non-empty values in {text_column!r}")

    scores = [measure_readability(text) for text in texts]
    return {
        "outputs_scored": len(scores),
        "percent_at_or_below_grade_8": (
            sum(score.flesch_kincaid_grade <= 8 for score in scores)
            / len(scores)
            * 100
        ),
        "average_flesch_kincaid_grade": fmean(
            score.flesch_kincaid_grade for score in scores
        ),
        "average_flesch_reading_ease": fmean(
            score.flesch_reading_ease for score in scores
        ),
        "average_smog_index": fmean(score.smog_index for score in scores),
        "average_coleman_liau_index": fmean(
            score.coleman_liau_index for score in scores
        ),
        "average_automated_readability_index": fmean(
            score.automated_readability_index for score in scores
        ),
        "average_gunning_fog_index": fmean(
            score.gunning_fog_index for score in scores
        ),
        "average_dale_chall_score": fmean(
            score.dale_chall_score for score in scores
        ),
        "average_difficult_word_ratio": fmean(
            score.difficult_word_ratio for score in scores
        ),
        "average_sentence_length": fmean(
            score.average_sentence_length for score in scores
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--text-column",
        default="simplified_transcription",
        help="CSV column containing generated text",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate_csv(args.csv_path, args.text_column), indent=2))


if __name__ == "__main__":
    main()
