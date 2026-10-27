"""
FaithfulMed — minimal Google ADK pipeline skeleton.

Shows the *shape* the SME review asked for: a deterministic DAG (SequentialAgent), not open-ended
recursion, with a single BOUNDED loop (LoopAgent, max_iterations=2) for Refiner <-> Verifier.

This is a scaffold to react to, not a finished pipeline. Each fellow fleshes out their own agent's
instruction, output schema, and tools.

Setup:
    pip install google-adk
    export GEMINI_API_KEY=...        # free key from https://aistudio.google.com
Docs: https://google.github.io/adk-docs

Note: ADK's exact class/kwarg names evolve — check the docs for your installed version. The intent
(SequentialAgent for the DAG, LoopAgent with a hard iteration cap for the refine step) is the point.
"""
import os

from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent

from agents.readability.agent import root_agent as readability

MODEL = os.getenv("FAITHFULMED_MODEL", "gemini-flash-latest")

# 1) Extractor — pull clinical "atoms" (facts + types) from the source document.
extractor = LlmAgent(
    name="Extractor",
    model=MODEL,
    instruction=(
        "Extract every clinically meaningful fact from the SOURCE as a JSON list of atoms, each "
        "with {text, type} where type is one of: diagnosis, medication, lab_value, procedure, "
        "instruction. Do not add facts that are not in the SOURCE."
    ),
    output_key="atoms",
)

# 2) Simplifier — rewrite into plain language, grounded by RAG (wire retrieval into instruction/tools).
simplifier = LlmAgent(
    name="Simplifier",
    model=MODEL,
    instruction=(
        "Rewrite the SOURCE for a patient at an 8th-grade reading level using only the extracted "
        "{atoms}. Use plain-language definitions retrieved from the glossary. Add nothing new."
    ),
    output_key="draft",
)

# 3) Verifier — LLM-as-judge: check the post-readability draft against {atoms}.
verifier = LlmAgent(
    name="Verifier",
    model=MODEL,
    instruction=(
        "Compare {readability_draft} against {atoms}. Return JSON: {faithful: bool, unsupported_claims: [...], "
        "omissions: [...], reading_level_ok: bool}. Be strict: an unsupported clinical claim is a failure."
    ),
    output_key="verdict",
)

# 4) Refiner — fix only what the Verifier flagged. No free rewriting.
refiner = LlmAgent(
    name="Refiner",
    model=MODEL,
    instruction=(
        "Given {readability_draft} and {verdict}, produce a corrected draft that removes unsupported "
        "claims and restores omissions, changing nothing else. If {verdict.faithful} is true, return "
        "{readability_draft} unchanged."
    ),
    output_key="readability_draft",
)

# Bounded refine loop: Verifier -> Refiner, at most twice. This is the ONLY loop in the system.
refine_loop = LoopAgent(
    name="RefineLoop",
    sub_agents=[verifier, refiner],
    max_iterations=2,
)

final_verifier = LlmAgent(
    name="FinalVerifier",
    model=MODEL,
    instruction=(
        "Perform a final check of {readability_draft} against {atoms} after refinement. "
        "Return JSON with faithful, unsupported_claims, omissions, and reading_level_ok. "
        "Do not rewrite the explanation."
    ),
    output_key="final_verdict",
)

# Verify after readability and again after the bounded refine loop's last possible rewrite.
faithfulmed = SequentialAgent(
    name="FaithfulMed",
    sub_agents=[extractor, simplifier, readability, refine_loop, final_verifier],
)

if __name__ == "__main__":
    print("FaithfulMed ADK pipeline assembled:")
    print(
        "  Extractor -> Simplifier -> Readability -> "
        "[Verifier <-> Refiner x<=2] -> FinalVerifier"
    )
    print("Run it with an ADK Runner + session (see the ADK quickstart) once GEMINI_API_KEY is set.")
