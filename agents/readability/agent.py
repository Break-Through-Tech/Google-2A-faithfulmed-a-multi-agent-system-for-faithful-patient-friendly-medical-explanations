"""Google ADK agent for patient-friendly readability revision."""

import os

from google.adk.agents import LlmAgent

from .observability import configure_cloud_observability

configure_cloud_observability()

MODEL = os.getenv("FAITHFULMED_MODEL", "gemini-flash-latest")

root_agent = LlmAgent(
    name="readability_agent",
    model=MODEL,
    description=(
        "Revises a patient-facing medical explanation for readability while preserving "
        "all source-supported clinical information."
    ),
    instruction=(
        "You are the FaithfulMed Readability Agent. Rewrite {draft?} for a patient using "
        "clear, direct language and a respectful, calm tone. Aim for a Flesch-Kincaid "
        "grade level of {target_grade?}, defaulting to grade 8 when no target is supplied.\n\n"
        "If no draft is supplied in workflow state, use the clinical text in the user's "
        "current request. If the request contains no clinical text, ask the user to provide it.\n\n"
        "Preserve every clinical fact in {atoms?} and every item in {protected_facts?}. "
        "Do not change or omit diagnoses, medication names, doses, frequencies, dates, "
        "measurements, lab values, negation, uncertainty, warnings, or follow-up instructions. "
        "Do not add advice, diagnoses, explanations, or facts that are not in the draft or "
        "the supplied clinical facts. Preserve the original meaning and uncertainty.\n\n"
        "If {glossary_context?} is supplied by the upstream retrieval step, use it only to "
        "choose a plain-language equivalent for a term. Do not add new medical information "
        "from the glossary. Return only the revised patient-facing text, without a preamble, "
        "grade-level claim, or evaluation notes."
    ),
    output_key="readability_draft",
)
