"""FaithfulMed pipeline: Extractor -> Simplifier -> [Verifier <-> Refiner x<=2]
(-> optional final Verifier check). The Refiner also handles readability.

build_pipeline() takes flags so the same code runs every ablation:
    use_extractor=False     -> Simplifier gets no atoms (falls back to sentence queries)
    use_refine_loop=False   -> no Verifier/Refiner loop
State flag use_rag=False (see run_pipeline) disables retrieval.
"""

import asyncio
import json
import time
from typing import Any, List, Optional

import textstat
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

try:
    from .extractor import EXTRACTOR_INSTRUCTIONS, MODEL as EXTRACTOR_MODEL, validate_output
    from .simplifier import (
        ATOMS_KEY, SimplifierOutput, TEMPERATURE, _user_text,
        prepare_verifier_input, retrieve_for_simplifier, simplifier_instruction,
    )
    from .verifier import MODEL, VERIFIER_INSTRUCTIONS, check_verifier_output
    from .refiner import make_refiner
    from .simplifier import _coerce_atoms, _grade
    from .verifier import make_verifier_input
except ImportError:  # plain script from inside agents/
    from extractor import EXTRACTOR_INSTRUCTIONS, MODEL as EXTRACTOR_MODEL, validate_output
    from simplifier import (
        ATOMS_KEY, SimplifierOutput, TEMPERATURE, _user_text,
        prepare_verifier_input, retrieve_for_simplifier, simplifier_instruction,
    )
    from verifier import MODEL, VERIFIER_INSTRUCTIONS, check_verifier_output
    from refiner import make_refiner
    from simplifier import _coerce_atoms, _grade
    from verifier import make_verifier_input

APP_NAME = "faithfulmed"
LLM_TIMEOUT_MS = 90_000  # fail instead of hanging forever on a stuck model call

# helping with 503 errors
def _http_options() -> types.HttpOptions:
    """Per-call timeout plus retries on overload/rate-limit errors."""
    return types.HttpOptions(
        timeout=LLM_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(
            attempts=5,            # includes the first try
            initial_delay=2.0,     # seconds
            max_delay=30.0,
            exp_base=2.0,
            http_status_codes=[429, 500, 503],
        ),
    )



def _load(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


# ---------------------------------------------------------------------------- Extractor

class ExtractedAtom(BaseModel):
    text: str
    type: str
    source_quote: str


class ExtractorOutput(BaseModel):
    atoms: List[ExtractedAtom] = Field(default_factory=list)
    not_found_types: List[str] = Field(default_factory=list)


def extractor_instruction(ctx: ReadonlyContext) -> str:
    note = ctx.state.get("original_text", "")
    return EXTRACTOR_INSTRUCTIONS + "\n\nINPUT JSON:\n" + json.dumps({"clinical_note": note})


def set_original_text(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    text = state.get("original_text") or _user_text(callback_context)
    if not text:
        raise ValueError("Pipeline needs original_text in state or in the user message.")
    state["original_text"] = text
    return None


def validate_extraction(callback_context: CallbackContext) -> Optional[types.Content]:
    """Reuse extractor.validate_output: drops atoms whose source_quote isn't in the note
    and recomputes not_found_types. Validated atoms become the Simplifier's clinical_atoms."""
    state = callback_context.state
    raw = _load(state.get("extractor_output"))
    if raw is None:
        return None  # call was cancelled or returned nothing
    result = validate_output(raw, state["original_text"])
    state[ATOMS_KEY] = result["atoms"]
    state["extraction_report"] = {
        k: result[k] for k in ("not_found_types", "rejected_atoms", "warnings")
    }
    return None


def make_extractor() -> LlmAgent:
    return LlmAgent(
        name="extractor_agent",
        model=EXTRACTOR_MODEL,
        description="Extracts atomic, source-quoted clinical facts.",
        instruction=extractor_instruction,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
            http_options=_http_options(),
        ),
        output_schema=ExtractorOutput,
        output_key="extractor_output",
        include_contents="none",
        before_agent_callback=set_original_text,
        after_agent_callback=validate_extraction,
    )


# ---------------------------------------------------------------------------- Simplifier

def after_simplifier(callback_context: CallbackContext) -> Optional[types.Content]:
    """Build verifier_input (simplifier.py) and expose the draft as state["simplified_text"]."""
    prepare_verifier_input(callback_context)
    out = _load(callback_context.state.get("simplified_output"))
    callback_context.state["simplified_text"] = out["simplified_text"]
    return None


def rebuild_after_refine(callback_context: CallbackContext) -> None:
    """Refiner overwrote simplified_text: re-point the Verifier at the new draft."""
    state = callback_context.state
    state["refine_rounds"] = state.get("refine_rounds", 0) + 1
    state["verifier_input"] = make_verifier_input(
        original_text=state.get("original_text", ""),
        simplified_text=state["simplified_text"],
        clinical_atoms=_coerce_atoms(state.get(ATOMS_KEY)),
        target_grade_level=_grade(state.get("target_grade_level")),
    )


def make_simplifier() -> LlmAgent:
    """Fresh instance each call (an ADK agent can only have one parent)."""
    return LlmAgent(
        name="simplifier_agent",
        model=MODEL,
        description="Rewrites clinical text in plain language, grounded in MedlinePlus.",
        instruction=simplifier_instruction,
        generate_content_config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            http_options=_http_options(),
        ),
        output_schema=SimplifierOutput,
        output_key="simplified_output",
        include_contents="none",
        before_agent_callback=retrieve_for_simplifier,
        after_agent_callback=after_simplifier,
    )


# ----------------------------------------------------------------------------- Verifier

def verifier_instruction(ctx: ReadonlyContext) -> str:
    """The Verifier expects JSON input; take it from state so it always sees the latest draft."""
    return VERIFIER_INSTRUCTIONS + "\n\nINPUT JSON:\n" + ctx.state.get("verifier_input", "")


def make_verifier(name: str, output_key: str, escalate_on_pass: bool) -> LlmAgent:
    try:
        from .verifier import VerifierOutput
    except ImportError:
        from verifier import VerifierOutput

    def after(callback_context: CallbackContext) -> Optional[types.Content]:
        raw = _load(callback_context.state.get(output_key))
        if raw is None:
            return None  # call was cancelled or returned nothing
        
        result = check_verifier_output(raw)
        state = callback_context.state
        state[output_key] = result.model_dump()
        fk = textstat.flesch_kincaid_grade(state.get("simplified_text", ""))
        if "fk_grade_before" not in state:
            state["fk_grade_before"] = fk  # grade of the first draft
        if not escalate_on_pass:
            state["fk_grade_after"] = fk  # final check: grade of the text actually shown
        if escalate_on_pass and result.overall_pass:
            # Exit the loop only if the Verifier passed AND the draft is on target;
            # otherwise the Refiner still has readability work to do.
            if fk <= _grade(state.get("target_grade_level")):
                callback_context.actions.escalate = True
        return None

    return LlmAgent(
        name=name,
        model=MODEL,
        description="Checks medical simplifications for accuracy and readability.",
        instruction=verifier_instruction,
        generate_content_config=types.GenerateContentConfig(
            http_options=_http_options(),
        ),
        output_schema=VerifierOutput,
        output_key=output_key,
        include_contents="none",
        after_agent_callback=after,
    )


# ----------------------------------------------------------------------------- Assembly

def build_pipeline(
    use_extractor: bool = True,
    use_refine_loop: bool = True,
    final_check: bool = True,
    max_iterations: int = 2,
) -> SequentialAgent:
    steps = []
    if use_extractor:
        steps.append(make_extractor())
    steps.append(make_simplifier())
    if use_refine_loop:
        steps.append(
            LoopAgent(
                name="refine_loop",
                sub_agents=[
                    make_verifier("loop_verifier", "verifier_feedback", escalate_on_pass=True),
                    make_refiner(on_refined=rebuild_after_refine),
                ],
                max_iterations=max_iterations,
            )
        )
    if final_check:
        # The loop's last Refiner output is never re-verified, and the Refiner's edits can drift:
        # one last Verifier pass scores the text that is actually shown. Not part of the loop.
        steps.append(make_verifier("final_verifier", "final_verifier_feedback", escalate_on_pass=False))
    return SequentialAgent(name="FaithfulMed", sub_agents=steps)


async def run_pipeline(
    note: str, grade: int = 8, use_rag: bool = True, **build_kwargs: Any
) -> dict:
    agent = build_pipeline(**build_kwargs)
    service = InMemorySessionService()
    session = await service.create_session(
        app_name=APP_NAME,
        user_id="eval",
        state={"original_text": note, "target_grade_level": grade, "use_rag": use_rag},
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=service)
    message = types.Content(role="user", parts=[types.Part(text=note)])

    t0 = time.time()
    async for event in runner.run_async(
        user_id="eval", session_id=session.id, new_message=message
    ):
        print(f"[{time.time() - t0:5.1f}s] {event.author}", flush=True)

    state = (
        await service.get_session(app_name=APP_NAME, user_id="eval", session_id=session.id)
    ).state
    simplified = state.get("simplified_text", "")
    return {
        "final_text": state.get("final_text") or simplified,
        "atoms": state.get(ATOMS_KEY, []),
        "sources_used": state.get("sources_used", []),
        "refine_rounds": state.get("refine_rounds", 0),
        "loop_verdict": state.get("verifier_feedback"),
        "final_verdict": state.get("final_verifier_feedback"),
        "fk_grade_before": state.get("fk_grade_before"),
        "fk_grade_after": state.get("fk_grade_after"),
        "state": state,  # full trace for the demo / error analysis
    }


if __name__ == "__main__":
    demo_note = (
        "The patient reports a dry cough for three days. No fever or shortness of breath "
        "is reported. A viral upper respiratory infection is suspected. The plan is rest, "
        "fluids, and follow-up if symptoms worsen or persist."
    )
    out = asyncio.run(run_pipeline(demo_note))
    print("FINAL TEXT:\n" + out["final_text"])
    print("\nREFINE ROUNDS:", out["refine_rounds"])
    print("FINAL VERDICT PASS:", (out["final_verdict"] or {}).get("overall_pass"))
    print("FK GRADE:", out["fk_grade_before"], "->", out["fk_grade_after"])
