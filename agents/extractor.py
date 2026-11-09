import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "data").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = PROJECT_ROOT / "data" / "mtsamples_baseline_outputs.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "extractor_outputs.jsonl"
MODEL = "gemini-3.5-flash"

ALLOWED_TYPES = {
    "diagnosis", "diagnosis_reasoning", "symptom", "risk", "medical_history",
    "family_history", "environmental_history", "demographic", "medical_specialty",
    "examination", "examination_finding", "lab_value", "vital_sign", "procedure",
    "treatment", "medication", "referral", "follow_up", "patient_education",
    "negative_finding",
}

EXTRACTOR_INSTRUCTIONS = """
You extract atomic clinical facts from a clinical note.

Input: JSON containing clinical_note.

Output: Return only valid JSON with exactly these fields:
{
  "atoms": [
    {"text": "...", "type": "...", "source_quote": "..."}
  ],
  "not_found_types": []
}

Extraction rules
-   Use only the source.
    Extract explicitly stated clinical facts without inferring, diagnosing, explaining, or adding information.
    Do not use outside knowledge.
-   Each atom must represent only one finding, symptom, diagnosis, medication, lab value, or other distinct fact.
    Split lists into separate atoms when each item is a distinct fact.
    Preserve qualifiers and relationships necessary to retain the original meaning after splitting.
-   source_quote: an exact contiguous quote from the clinical note that provides enough context to support the complete atom.
    Do not use an isolated phrase as the source_quote when the surrounding text is needed to establish its meaning or create a complete sentence.
-   Text should be a concise statement using only minimal grammatical normalization.
    Do not expand abbreviations, correct typos, or change wording.
-   Preserve status and details.
    Retain all negation, uncertainty, temporality, status, numbers, units, doses, routes, frequencies, and other qualifiers.
    Never assume unstated details.
-   Refer to the individual as "patient" in the extracted text.
    Do not extract names, dates of birth, addresses, phone numbers, or other identifiers.
-   Assign one allowed category per atom, choosing the most appropriate category:
    - diagnosis
    - diagnosis_reasoning
    - symptom
    - risk
    - medical_history
    - family_history
    - environmental_history
    - demographic
    - medical_specialty
    - examination
    - examination_finding
    - lab_value
    - vital_sign
    - procedure
    - treatment
    - medication
    - referral
    - follow_up
    - patient_education
    - negative_finding
-   Report missing categories.
    not_found_types must list every allowed category that has no supporting text in the note.
    Do not create atoms for missing or unknown information.
-   Return JSON only.
    If no facts are found, return an empty atoms array and list all categories in not_found_types.
-   The source_quote must be copied exactly from the note.
-   The text may use minimal grammatical normalization for clarity.
-   Validate traceability using source_quote, not text.
-   Preserve family relationships exactly. "Father's side" is not the same as "father."
-   Preserve negation.
-   Preserve if a medication is indicated to be existing, stopped, new
-   Use negative_finding for explicit denials and absent findings.

Example
Source: "Significant for high blood pressure and diabetes as well as cancer on her father side"
Extract separate atoms for high blood pressure, diabetes, and cancer on the patient's father's side. Preserve the family relationship in the cancer atom.

The clinical note is provided in the input JSON as clinical_note.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "atoms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "type": {"type": "STRING", "enum": sorted(ALLOWED_TYPES)},
                    "source_quote": {"type": "STRING"},
                },
                "required": ["text", "type", "source_quote"],
            },
        },
        "not_found_types": {
            "type": "ARRAY",
            "items": {"type": "STRING", "enum": sorted(ALLOWED_TYPES)},
        },
    },
    "required": ["atoms", "not_found_types"],
}

def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def quote_is_in_source(source_text: str, source_quote: str) -> bool:
    if not isinstance(source_text, str) or not isinstance(source_quote, str):
        return False
    if not source_quote.strip():
        return False
    return source_quote in source_text or normalize_whitespace(source_quote) in normalize_whitespace(source_text)


def validate_output(model_output: dict[str, Any], source_text: str) -> dict[str, Any]:
    # Validate structure
    if not isinstance(model_output, dict):
        raise ValueError("Extractor response must be a JSON object.")
    atoms = model_output.get("atoms")
    not_found_types = model_output.get("not_found_types")
    if not isinstance(atoms, list) or not isinstance(not_found_types, list):
        raise ValueError("Extractor response must contain atoms and not_found_types lists.")

    validated_atoms = []
    rejected_atoms = []
    for index, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            rejected_atoms.append({
                "atom_index": index,
                "atom": atom,
                "reason": "atom_is_not_an_object",
            })
            continue
        required = {"text", "type", "source_quote"}
        missing = required - atom.keys()
        if missing:
            rejected_atoms.append({
                "atom_index": index,
                "atom": atom,
                "reason": "missing_required_fields",
                "details": {"missing_fields": sorted(missing)},
            })
            continue
        if atom["type"] not in ALLOWED_TYPES:
            rejected_atoms.append({
                "atom_index": index,
                "atom": atom,
                "reason": "invalid_type",
                "details": {"type": atom["type"]},
            })
            continue
        if not all(
            isinstance(atom[field], str) and atom[field].strip()
            for field in required
        ):
            rejected_atoms.append({
                "atom_index": index,
                "atom": atom,
                "reason": "empty_or_non_string_field",
            })
            continue

        if not quote_is_in_source(source_text, atom["source_quote"]):
            rejected_atoms.append({
                "atom_index": index,
                "atom": atom,
                "reason": "source_quote_not_found",
                "annotation": (
                    "The atom was rejected because its source_quote does not "
                    "appear in the clinical note, so the claim cannot be traced "
                    "to the provided source."
                ),
            })
            continue
        validated_atoms.append({field: atom[field].strip() for field in required})

    # Recalculate this from accepted atoms.
    present_types = {atom["type"] for atom in validated_atoms}
    calculated_not_found = sorted(ALLOWED_TYPES - present_types)
    warnings = []
    invalid_not_found = set(not_found_types) - ALLOWED_TYPES
    if invalid_not_found or set(not_found_types) != set(calculated_not_found):
        warnings.append({
            "reason": "inconsistent_not_found_types",
            "annotation": (
                "The model-supplied not_found_types did not match the accepted "
                "atoms, so the list was recalculated."
            ),
        })
    return {
        "atoms": validated_atoms,
        "not_found_types": calculated_not_found,
        "rejected_atoms": rejected_atoms,
        "warnings": warnings,
    }


def load_records(path: Path = DATA_PATH) -> pd.DataFrame:
    records = pd.read_csv(path)
    required_columns = {"sample_name", "transcription"}
    missing_columns = required_columns - set(records.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    records = records[["sample_name", "transcription"]].dropna()
    return records[records["transcription"].str.strip().ne("")].reset_index(drop=True)

def make_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before running Gemini extraction.")
    if genai is None or types is None:
        raise ImportError("Install the existing google-genai dependency first.")
    return genai.Client(api_key=api_key)

def extract_atoms(
    source_text: str,
    client: Any | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    if client is None:
        client = make_client()
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    input_json = "\n\nINPUT JSON:\n" + json.dumps({"clinical_note": source_text})
    prompt = EXTRACTOR_INSTRUCTIONS + input_json
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            if not response.text:
                raise ValueError("Gemini returned an empty response.")
            model_output = json.loads(response.text)
            return validate_output(model_output, source_text)
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            if attempt == max_retries:
                break
            prompt = (
                EXTRACTOR_INSTRUCTIONS
                + "\n\nCORRECTION: Return the required JSON object. "
                "Use only source_quote values copied from the clinical note."
                + input_json
            )
    raise RuntimeError(
        f"Extractor response remained invalid after {max_retries + 1} attempts: "
        f"{last_error}"
    ) from last_error

def run_extraction(
    records_to_process: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
    limit: int = 2,
) -> None:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    client = make_client()
    selected = records_to_process.head(limit)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in selected.itertuples(index=False):
            try:
                extraction = extract_atoms(row.transcription, client=client)
                result = {
                    "sample_name": row.sample_name,
                    "status": "success",
                    **extraction,
                }
            except Exception as error:
                result = {
                    "sample_name": row.sample_name,
                    "status": "failed",
                    "atoms": [],
                    "not_found_types": sorted(ALLOWED_TYPES),
                    "rejected_atoms": [],
                    "warnings": [],
                    "error": {
                        "type": type(error).__name__,
                        "annotation": str(error),
                    },
                }
            output_file.write(json.dumps(result) + "\n")
    print(f"Saved {len(selected)} extraction results to {output_path}")


if __name__ == "__main__":
    records = load_records()
    sample_source = records.iloc[1]["transcription"]
    sample_output = extract_atoms(sample_source)
    print(json.dumps(sample_output, indent=2))
    
    # run_extraction(records)