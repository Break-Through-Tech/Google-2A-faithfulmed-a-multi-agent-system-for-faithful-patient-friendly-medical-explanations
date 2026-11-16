"""Refiner agent: applies the fixes the Verifier flagged AND checks readability itself.

Readability is merged in here. Besides the Verifier's feedback, the Refiner measures the
draft's Flesch-Kincaid grade with textstat. It runs when EITHER the Verifier failed OR the
measured grade is above target, and it skips the LLM call only when both are fine.

Works standalone (refinement_loop.py) and inside the ADK pipeline (pipeline.py).
State in:  original_text, simplified_text, verifier_feedback, target_grade_level (default 8)
State out: simplified_text (overwritten), final_text, fk_grade_before, fk_grade_after
"""

import json
import re
from typing import Any, Callable, Optional

import textstat
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types

MODEL = "gemini-3.5-flash"  # confirm this model ID is available to your API key

# Placeholders: {original_text} {simplified_text} {verifier_feedback} {fk_grade} {target_grade}.
# Filled with str.format in refiner_instruction(), not ADK state templating, so braces in
# clinical text can never break it. Do not add other braces to this string.
REFINER_INSTRUCTION = """
Original medical text:
{original_text}

Current simplified explanation:
{simplified_text}

Verifier feedback:
{verifier_feedback}

Measured reading level of the current explanation (Flesch-Kincaid grade): {fk_grade}
Target: grade {target_grade} or below.

If overall_pass is true AND the measured grade is at or below the target,
return the simplified text unchanged.

Otherwise:
- Correct inaccuracies identified in faithfulness.
- Restore important information identified in omissions.
- Remove claims identified in unsupported_additions.
- Improve readability based on reading_level.
- Follow refiner_instructions when supported by the original text.
- If the measured grade is above the target, lower it by splitting long
  sentences and replacing hard words with everyday words. Change wording and
  sentence length only: keep every fact, number, unit, dose, negation and
  statement of certainty exactly as it is. Keep a drug or test name if the
  patient will hear it again, with its plain meaning in parentheses.

The original medical text is the authoritative source
for patient-specific facts.

Make minimal corrections and preserve accurate wording.

Treat the current simplified explanation as a draft to edit,
not an invitation to write a new explanation.

Make only the changes necessary to address the verifier's
identified issues and the reading-level target. Preserve all
other accurate content.

Do not add background information or explanations unless
they are necessary to correct an identified issue and are
supported by the original medical text.

Never add unsupported medical information.
Use patient-friendly language at or below an eighth-grade reading level.
Plain text paragraphs only: no markdown, headings or bullet symbols.

Output only the corrected explanation.
"""


def _load(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _grade(value: Any) -> int:
    try:
        return max(1, min(12, int(value)))
    except (TypeError, ValueError):
        return 8


def _clean(text: str) -> str:
    """Strip whitespace and any stray markdown code fence around the output."""
    text = text.strip()
    text = re.sub(r"^```\w*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def measure_grade(state) -> tuple[float, int]:
    """(Flesch-Kincaid grade of the current draft, target grade)."""
    text = state.get("simplified_text") or ""
    fk = textstat.flesch_kincaid_grade(text) if text.strip() else 0.0
    return fk, _grade(state.get("target_grade_level"))


def needs_refinement(state) -> bool:
    feedback = _load(state.get("verifier_feedback"))
    if not feedback:
        return False
    fk, target = measure_grade(state)
    return (not feedback.get("overall_pass")) or fk > target


def refiner_instruction(ctx: ReadonlyContext) -> str:
    s = ctx.state
    fk, target = measure_grade(s)
    return REFINER_INSTRUCTION.format(
        original_text=s.get("original_text", ""),
        simplified_text=s.get("simplified_text", ""),
        verifier_feedback=json.dumps(_load(s.get("verifier_feedback")) or {}, indent=2),
        fk_grade=f"{fk:.1f}",
        target_grade=target,
    )


def skip_if_fine(callback_context: CallbackContext) -> Optional[types.Content]:
    """before_agent_callback: Verifier passed AND grade on target -> skip the LLM call."""
    state = callback_context.state
    fk, _ = measure_grade(state)

    if "fk_grade_before" not in state:
        state["fk_grade_before"] = fk

    state["fk_grade_after"] = fk
    if needs_refinement(state):
        state["refiner_skipped"] = False
        return None

    state["refiner_skipped"] = True
    text = state.get("simplified_text", "")
    state["final_text"] = text
    return types.Content(role="model", parts=[types.Part(text=text)])

def make_refiner(
    on_refined: Optional[Callable[[CallbackContext], None]] = None,
) -> LlmAgent:
    """Fresh agent per call (an ADK agent can only have one parent).
    on_refined runs after a real refinement, e.g. to rebuild the Verifier's input."""

    def after(callback_context: CallbackContext) -> Optional[types.Content]:
        state = callback_context.state
        if state.get("refiner_skipped"):
            return None
        text = _clean(state.get("simplified_text") or "")
        if not text:
            raise RuntimeError("Refiner produced no text.")
        state["simplified_text"] = text
        state["final_text"] = text
        state["fk_grade_after"], _ = measure_grade(state)
        if on_refined:
            on_refined(callback_context)
        return None

    return LlmAgent(
        name="Refiner",
        model=MODEL,
        description="Fixes Verifier-flagged issues and brings the text to the target reading level.",
        instruction=refiner_instruction,
        output_key="simplified_text",
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                timeout=90_000,
                retry_options=types.HttpRetryOptions(
                    attempts=5,
                    initial_delay=2.0,
                    max_delay=30.0,
                    exp_base=2.0,
                    http_status_codes=[429, 500, 503],
                ),
            ),
        ),
        include_contents="none",
        before_agent_callback=skip_if_fine,
        after_agent_callback=after,
    )


refiner = make_refiner()  # standalone instance for refinement_loop.py
