"""Deterministic readability scoring and clinical-fact preservation checks."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, cast

import textstat
from opentelemetry import trace

logger = logging.getLogger("faithfulmed.readability")
tracer = trace.get_tracer("faithfulmed.readability")


class _TextstatAPI(Protocol):
    def lexicon_count(self, text: str, removepunct: bool = False) -> int: ...

    def sentence_count(self, text: str) -> int: ...

    def difficult_words(self, text: str) -> int: ...

    def flesch_reading_ease(self, text: str) -> float: ...

    def flesch_kincaid_grade(self, text: str) -> float: ...

    def smog_index(self, text: str) -> float: ...

    def coleman_liau_index(self, text: str) -> float: ...

    def automated_readability_index(self, text: str) -> float: ...

    def gunning_fog(self, text: str) -> float: ...

    def dale_chall_readability_score(self, text: str) -> float: ...


_textstat = cast(_TextstatAPI, textstat)


@dataclass(frozen=True)
class ReadabilityMetrics:
    word_count: int
    sentence_count: int
    average_sentence_length: float
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    smog_index: float
    coleman_liau_index: float
    automated_readability_index: float
    gunning_fog_index: float
    dale_chall_score: float
    difficult_word_ratio: float


@dataclass(frozen=True)
class ReadabilityAssessment:
    output_text: str
    candidate_metrics: ReadabilityMetrics
    status: Literal["preservation_failed", "accepted", "target_not_met"]
    meets_grade_target: bool
    missing_protected_facts: tuple[str, ...]

    @property
    def preservation_passed(self) -> bool:
        return not self.missing_protected_facts


def measure_readability(text: str) -> ReadabilityMetrics:
    """Calculate established readability measures for non-empty text.

    Flesch-Kincaid grade is the project's primary threshold measure. Other formulas
    are reported as complementary indicators, not treated as interchangeable grades.
    Difficult-word ratio is a general vocabulary proxy, not a medical-jargon detector.
    """
    if not text.strip():
        raise ValueError("text must contain non-whitespace characters")

    word_count = max(_textstat.lexicon_count(text, removepunct=True), 1)
    sentence_count = max(_textstat.sentence_count(text), 1)
    difficult_word_count = _textstat.difficult_words(text)

    return ReadabilityMetrics(
        word_count=word_count,
        sentence_count=sentence_count,
        average_sentence_length=word_count / sentence_count,
        flesch_reading_ease=_textstat.flesch_reading_ease(text),
        flesch_kincaid_grade=_textstat.flesch_kincaid_grade(text),
        smog_index=_textstat.smog_index(text),
        coleman_liau_index=_textstat.coleman_liau_index(text),
        automated_readability_index=_textstat.automated_readability_index(text),
        gunning_fog_index=_textstat.gunning_fog(text),
        dale_chall_score=_textstat.dale_chall_readability_score(text),
        difficult_word_ratio=difficult_word_count / word_count,
    )


def _normalize_for_matching(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def assess_candidate(
    original_draft: str,
    candidate: str,
    protected_facts: list[str] | tuple[str, ...] = (),
    target_grade: float = 8.0,
) -> ReadabilityAssessment:
    """Evaluate a rewrite and fall back to the original if protected facts were lost.

    Call this after the ADK agent. A non-passing status must remain visible to the
    caller; the function never treats a failed rewrite as an accepted simplification.
    """
    if not original_draft.strip():
        raise ValueError("original_draft must contain non-whitespace characters")
    if not candidate.strip():
        raise ValueError("candidate must contain non-whitespace characters")
    if not math.isfinite(target_grade) or target_grade <= 0:
        raise ValueError("target_grade must be a finite number greater than zero")

    normalized_candidate = _normalize_for_matching(candidate)
    missing_facts = tuple(
        fact for fact in protected_facts
        if not fact.strip() or _normalize_for_matching(fact) not in normalized_candidate
    )
    metrics = measure_readability(candidate)
    meets_target = metrics.flesch_kincaid_grade <= target_grade

    if missing_facts:
        status = "preservation_failed"
        output_text = original_draft
    elif meets_target:
        status = "accepted"
        output_text = candidate
    else:
        status = "target_not_met"
        output_text = candidate

    assessment = ReadabilityAssessment(
        output_text=output_text,
        candidate_metrics=metrics,
        status=status,
        meets_grade_target=meets_target,
        missing_protected_facts=missing_facts,
    )
    _record_assessment(assessment, target_grade)
    return assessment


def _record_assessment(assessment: ReadabilityAssessment, target_grade: float) -> None:
    metrics = asdict(assessment.candidate_metrics)
    log_fields = {
        "event": "faithfulmed.readability.assessed",
        "status": assessment.status,
        "target_grade": target_grade,
        "meets_grade_target": assessment.meets_grade_target,
        "preservation_passed": assessment.preservation_passed,
        "missing_protected_fact_count": len(assessment.missing_protected_facts),
        **metrics,
    }

    with tracer.start_as_current_span("faithfulmed.readability.assess") as span:
        for name, value in log_fields.items():
            if isinstance(value, (bool, int, float, str)):
                span.set_attribute(name, value)
        logger.info(json.dumps(log_fields, separators=(",", ":")))
