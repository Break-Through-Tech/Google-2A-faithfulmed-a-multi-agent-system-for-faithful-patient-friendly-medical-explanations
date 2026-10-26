"""Verifier agent for checking simplified medical text."""

from typing import List, Literal, Optional

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field, field_validator


MODEL = "gemini-3-flash-preview"


class ClinicalAtom(BaseModel):
    id: str
    concept: str
    importance: Optional[str] = None


class VerifierInput(BaseModel):
    original_text: str
    simplified_text: str
    clinical_atoms: List[ClinicalAtom] = Field(default_factory=list)
    target_grade_level: int = Field(default=8, ge=1, le=12)

    @field_validator("original_text", "simplified_text")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be empty")
        return value.strip()


class Issue(BaseModel):
    description: str
    severity: Literal["low", "medium", "high"]
    source_evidence: Optional[str] = None
    simplified_evidence: Optional[str] = None
    atom_id: Optional[str] = None


class CheckResult(BaseModel):
    passed: bool
    issues: List[Issue] = Field(default_factory=list)


class ReadingLevelResult(BaseModel):
    passed: bool
    estimated_grade: Optional[float] = None
    issues: List[Issue] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    faithfulness: CheckResult
    omissions: CheckResult
    unsupported_additions: CheckResult
    reading_level: ReadingLevelResult
    overall_pass: bool
    refiner_instructions: List[str] = Field(default_factory=list)


def make_verifier_input(
    original_text: str,
    simplified_text: str,
    clinical_atoms: Optional[List[dict]] = None,
    target_grade_level: int = 8,
) -> str:
    """Validate and format an input for the verifier."""
    data = VerifierInput(
        original_text=original_text,
        simplified_text=simplified_text,
        clinical_atoms=clinical_atoms or [],
        target_grade_level=target_grade_level,
    )
    return data.model_dump_json(indent=2)


def check_verifier_output(response: dict) -> VerifierOutput:
    """Validate feedback and keep the pass flags consistent with the issues."""
    result = VerifierOutput.model_validate(response)
    checks = [
        result.faithfulness,
        result.omissions,
        result.unsupported_additions,
        result.reading_level,
    ]
    for check in checks:
        if check.issues:
            check.passed = False
    result.overall_pass = all(check.passed for check in checks)
    return result


VERIFIER_INSTRUCTIONS = """
You are checking whether a simplified medical explanation accurately represents
its original source text. The input is JSON containing original_text,
simplified_text, clinical_atoms (optional), and target_grade_level.

Evaluate four things:
1. Faithfulness: Are the simplified claims consistent with the source? Check
   changes to certainty, negation, numbers, timing, and clinical meaning.
2. Omissions: Is any clinically important information missing? Do not flag
   technical details that can be safely omitted for a patient-facing summary.
3. Unsupported additions: Does the simplification add patient-specific facts,
   diagnoses, risks, or treatment advice not supported by the source?
4. Reading level: Is the wording understandable at the target grade level?
   Give an approximate grade estimate; do not present it as a measured score.

Treat original_text as the source of truth. clinical_atoms can help find
omissions, but verify them against original_text before using them. Do not
assume the extracted atoms are complete or correct. General definitions are
acceptable only when they do not introduce unsupported clinical claims.

For each issue, give a short description and severity (low, medium, high).
Include a brief source quote when relevant, and the problematic simplified
phrase when available. Use atom_id only if the issue maps to a supplied atom.
Do not invent evidence quotes.

Set each check's passed field to false when it has issues that need correction.
Do not flag minor stylistic suggestions as issues unless they affect the target reading level.
Use high severity for dangerous changes to medications, dosages, negation, or urgent warnings;
medium for other clinically important inaccuracies or omissions; low for minor clarity issues.
Set overall_pass to true only if all four checks pass. If overall_pass is false,
provide specific refiner_instructions describing what to change. Preserve
accurate information and avoid adding medical advice that was not in the source.
Return only the structured output. Do not include markdown fences.
"""


verifier_agent = LlmAgent(
    name="verifier_agent",
    model=MODEL,
    description="Checks medical simplifications for accuracy and readability.",
    instruction=VERIFIER_INSTRUCTIONS,
    output_schema=VerifierOutput,
    output_key="verifier_feedback",
)
