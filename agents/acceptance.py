"""Conservative deterministic checks and review status for generated text."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_NUMBER_WORDS = {
    "a": "1",
    "an": "1",
    "once": "1",
    "twice": "2",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
_QUANTITY_RE = re.compile(
    r"\b(?P<number>\d+(?:\.\d+)?|"
    + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    + r")\s*(?P<unit>milligrams?|mg|micrograms?|mcg|"
    r"millilit(?:er|re)s?|ml|grams?|g|kilograms?|kg|"
    r"days?|weeks?|months?|years?|hours?|hrs?|minutes?|mins?|"
    r"times?|daily|weekly|monthly|years?|percent|%)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|denies?|without|negative for|absent)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(?:suspected|suspect|possible|possibly|may|might|could|"
    r"likely|unlikely|probable|possibly|rule out|cannot exclude)\b",
    re.IGNORECASE,
)
_REPORTING_RE = re.compile(
    r"\b(?:reports?|reported|denies?|denied|states?|stated)\b",
    re.IGNORECASE,
)
_CONFIRMATION_RE = re.compile(
    r"\b(?:confirmed|confirmatory|definitive|definitely|diagnosed with)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _quantities(text: str) -> set[str]:
    quantities: set[str] = set()
    for match in _QUANTITY_RE.finditer(text):
        raw_number = match.group("number").casefold()
        number = _NUMBER_WORDS.get(raw_number, raw_number)
        unit = match.group("unit").casefold()
        if unit.endswith("s") and unit not in {"hrs", "mins"}:
            unit = unit[:-1]
        quantities.add(f"{number} {unit}")
    quantities.update(
        f"numeric-value {match.group()}"
        for match in re.finditer(r"\b\d+(?:\.\d+)?\b", text)
    )
    return quantities


def assess_acceptance(
    source_text: str,
    candidate_text: str,
    verifier_feedback: Mapping[str, Any],
    protected_facts: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return safety signals; this prototype never auto-approves medical text.

    Checks are intentionally conservative. A flagged phrase requires human
    review; an unflagged phrase is not proof of clinical equivalence.
    """
    source = _normalize(source_text)
    candidate = _normalize(candidate_text)
    issues: list[dict[str, str]] = []

    for fact in protected_facts:
        normalized_fact = _normalize(fact)
        if not normalized_fact or normalized_fact not in candidate:
            issues.append({
                "code": "protected_fact_missing",
                "detail": f"User-marked critical fact is not preserved exactly: {fact}",
            })

    missing_quantities = sorted(_quantities(source) - _quantities(candidate))
    for quantity in missing_quantities:
        issues.append({
            "code": "quantity_or_timing_changed",
            "detail": f"Source quantity, dose, or timing value is missing or changed: {quantity}",
        })

    if _NEGATION_RE.search(source) and not _NEGATION_RE.search(candidate):
        issues.append({
            "code": "negation_not_preserved",
            "detail": "Source contains negation, but candidate has no recognized negation marker.",
        })

    if _UNCERTAINTY_RE.search(source) and not _UNCERTAINTY_RE.search(candidate):
        issues.append({
            "code": "uncertainty_not_preserved",
            "detail": "Source expresses uncertainty, but candidate has no recognized uncertainty marker.",
        })
    if _UNCERTAINTY_RE.search(source) and _CONFIRMATION_RE.search(candidate):
        issues.append({
            "code": "certainty_strengthened",
            "detail": "Candidate uses confirmed/definitive wording while source expresses uncertainty.",
        })

    if (
        re.search(
            r"\bis reported\b|\bare reported\b|\bwas reported\b|"
            r"\breports?\s+(?:that\s+)?(?:no|not|never)\b",
            source,
        )
        and not _REPORTING_RE.search(candidate)
    ):
        issues.append({
            "code": "reported_status_not_preserved",
            "detail": (
                "Source attributes information as reported or patient-stated, "
                "but candidate has no recognized reporting attribution."
            ),
        })

    verifier_issues: list[dict[str, str]] = []
    for category in (
        "faithfulness",
        "omissions",
        "unsupported_additions",
        "reading_level",
    ):
        check = verifier_feedback.get(category, {})
        if isinstance(check, Mapping):
            category_issues = check.get("issues", [])
            if isinstance(category_issues, list):
                for issue in category_issues:
                    if isinstance(issue, Mapping):
                        verifier_issues.append({
                            "category": category,
                            "severity": str(issue.get("severity", "unknown")),
                            "detail": str(issue.get("description", "Verifier flagged an issue.")),
                        })

    review_reasons = [item["detail"] for item in issues]
    review_reasons.extend(
        f"Verifier {item['category']} ({item['severity']}): {item['detail']}"
        for item in verifier_issues
    )
    if not review_reasons:
        review_reasons.append(
            "The LLM Verifier passing is not proof of clinical correctness."
        )

    deterministic_passed = not issues
    verifier_passed = bool(verifier_feedback.get("overall_pass", False))
    return {
        "review_status": "human_review_required",
        "deterministic_checks_passed": deterministic_passed,
        "deterministic_issues": issues,
        "verifier_passed": verifier_passed,
        "verifier_issues": verifier_issues,
        "review_reasons": review_reasons,
    }
