from google.adk.agents import LlmAgent

MODEL = "gemini-3.8-flash"

REFINER_INSTRUCTION = """
Original medical text:
{original_text}

Current simplified explanation:
{simplified_text}

Verifier feedback:
{verifier_feedback}

If overall_pass is true, return the simplified text unchanged.

Otherwise:
- Correct inaccuracies identified in faithfulness.
- Restore important information identified in omissions.
- Remove claims identified in unsupported_additions.
- Improve readability based on reading_level.
- Follow refiner_instructions when supported by the original text.

The original medical text is the authoritative source
for patient-specific facts.

Make minimal corrections and preserve accurate wording.

Treat the current simplified explanation as a draft to edit,
not an invitation to write a new explanation.

Make only the changes necessary to address the verifier's
identified issues. Preserve all other accurate content.

Do not add background information or explanations unless
they are necessary to correct an identified issue and are
supported by the original medical text.

Never add unsupported medical information.
Use patient-friendly language at or below an eighth-grade reading level.

Output only the corrected explanation.
"""

refiner = LlmAgent(
    name="Refiner",
    model=MODEL,
    instruction=REFINER_INSTRUCTION,
    output_key="simplified_text",
)