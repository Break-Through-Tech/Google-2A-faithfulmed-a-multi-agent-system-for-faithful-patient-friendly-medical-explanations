"""Direct Gemini agent for extracting source-grounded clinical atoms."""

import re
from typing import Literal

from pydantic import BaseModel, Field

from agents.model_runner import AgentDefinition


AtomType = Literal[
    "diagnosis",
    "diagnosis_reasoning",
    "symptom",
    "negative_finding",
    "risk",
    "medical_history",
    "family_history",
    "environmental_history",
    "demographic",
    "medical_specialty",
    "examination",
    "examination_finding",
    "lab_value",
    "vital_sign",
    "procedure",
    "treatment",
    "medication",
    "referral",
    "follow_up",
    "patient_education",
]

ALLOWED_TYPES = set(AtomType.__args__)


class ExtractedAtom(BaseModel):
    text: str
    type: AtomType
    source_quote: str


class ExtractorOutput(BaseModel):
    atoms: list[ExtractedAtom] = Field(default_factory=list)
    not_found_types: list[AtomType] = Field(default_factory=list)


EXTRACTOR_INSTRUCTIONS = """
Extract atomic clinical facts from the source clinical note in the input JSON.
Return the structured output only. Each atom must contain one explicitly stated
fact, one category, and an exact contiguous source_quote supporting the full
fact. Do not infer, diagnose, explain, expand abbreviations, correct typos, or
use outside knowledge. Keep text concise with only minimal grammatical
normalization. Do not extract personal identifiers.

Preserve negation, uncertainty, temporality, status, numbers, units, doses,
routes, frequencies, relationships, and qualifiers. Split distinct facts into
separate atoms while retaining the context needed to keep their meaning.
Use negative_finding for explicit denials and absent findings.

not_found_types must list every allowed category that has no supporting source
text. Never create an atom for missing information. Use only these categories:
diagnosis, diagnosis_reasoning, symptom, negative_finding, risk,
medical_history, family_history, environmental_history, demographic,
medical_specialty, examination, examination_finding, lab_value, vital_sign,
procedure, treatment, medication, referral, follow_up, patient_education.
"""

extractor_agent = AgentDefinition(
    name="extractor_agent",
    instruction=EXTRACTOR_INSTRUCTIONS,
    output_key="extractor_output",
    output_schema=ExtractorOutput,
)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def validate_extractor_output(
    model_output: ExtractorOutput | dict[str, object],
    source_text: str,
) -> ExtractorOutput:
    """Validate extractor categories and source-quote traceability."""
    if not source_text.strip():
        raise ValueError("Source clinical note cannot be empty.")
    result = (
        model_output
        if isinstance(model_output, ExtractorOutput)
        else ExtractorOutput.model_validate(model_output)
    )
    normalized_source = _normalize_whitespace(source_text)
    present_types: set[str] = set()
    for index, atom in enumerate(result.atoms, start=1):
        quote = _normalize_whitespace(atom.source_quote)
        if not quote or quote not in normalized_source:
            raise ValueError(
                f"Extractor atom {index} source_quote is not present in the source note."
            )
        if not atom.text.strip():
            raise ValueError(f"Extractor atom {index} has empty text.")
        present_types.add(atom.type)

    if len(result.not_found_types) != len(set(result.not_found_types)):
        raise ValueError("Extractor not_found_types contains duplicate categories.")
    if set(result.not_found_types) != ALLOWED_TYPES - present_types:
        raise ValueError(
            "Extractor not_found_types must list exactly the categories absent "
            "from the extracted atoms."
        )
    return result
