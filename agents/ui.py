"""Local Gradio interface for the multi-agent FaithfulMed ADK prototype."""

import csv
import json
import os
from pathlib import Path
from typing import Mapping

import gradio as gr

from agents.contracts import PipelineCase
from agents.run_pipeline import (
    MAX_REFINEMENTS,
    SYNTHETIC_SOURCE,
    run_pipeline,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_LIMIT = 8

DEFAULT_PROTECTED_FACTS = "\n".join(
    [
        "three days",
        "No fever",
        "No fever or shortness of breath is reported",
        "is suspected",
        "rest, fluids",
        "follow-up if symptoms worsen or persist",
    ]
)


def load_dataset_examples(
    data_dir: Path = DATA_DIR,
    limit_per_dataset: int = DATASET_LIMIT,
) -> dict[str, str]:
    """Load a small catalog of longer local dataset records for UI testing."""
    if limit_per_dataset < 1:
        raise ValueError("limit_per_dataset must be at least one")

    examples: dict[str, str] = {}
    mtsamples_path = data_dir / "mtsamples_baseline_outputs.csv"
    if mtsamples_path.is_file():
        with mtsamples_path.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"sample_name", "transcription"}
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"{mtsamples_path} is missing columns: {sorted(missing_columns)}"
                )
            records = [
                row
                for row in reader
                if (row.get("transcription") or "").strip()
            ]
        records.sort(
            key=lambda row: len(row["transcription"]),
            reverse=True,
        )
        for row in records[:limit_per_dataset]:
            name = row["sample_name"].strip() or "Unnamed sample"
            source = row["transcription"].strip()
            label = (
                f"MTSamples | {name} | {len(source)} chars / "
                f"{len(source.split())} words"
            )
            examples[label] = source

    medaesqa_path = data_dir / "medaesqa_v1.json"
    if medaesqa_path.is_file():
        records = json.loads(medaesqa_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{medaesqa_path} must contain a JSON list")
        medaesqa_records = [
            row
            for row in records
            if isinstance(row, dict)
            and isinstance(row.get("expert_curated_answer"), str)
            and row["expert_curated_answer"].strip()
        ]
        medaesqa_records.sort(
            key=lambda row: len(row["expert_curated_answer"]),
            reverse=True,
        )
        for row in medaesqa_records[:limit_per_dataset]:
            answer = row["expert_curated_answer"].strip()
            question = " ".join(str(row.get("question", "")).split())
            label = (
                f"MedAESQA | ID {row.get('question_id', 'unknown')} | "
                f"{question[:100]} | {len(answer)} chars"
            )
            examples[label] = answer

    return examples


def load_selected_dataset_case(
    selection: str,
    examples: Mapping[str, str],
) -> tuple[str, str]:
    """Load a selected dataset record into the source field and clear facts."""
    if selection not in examples:
        raise ValueError("Select an available local dataset example.")
    return examples[selection], ""


def build_case_from_ui(
    source_text: str,
    protected_facts_text: str,
    target_grade: int,
) -> PipelineCase:
    """Build the shared case from user-entered source and optional exact facts."""
    facts = [
        line.strip()
        for line in protected_facts_text.splitlines()
        if line.strip()
    ]
    extractor_output = {
        "atoms": [
            {
                "text": fact,
                "type": "user_supplied_fact",
                "source_quote": fact,
            }
            for fact in facts
        ]
    }
    return PipelineCase.from_extractor_output(
        source_text=source_text,
        extractor_output=extractor_output,
        protected_facts=facts,
        target_grade=target_grade,
    )


async def run_ui_pipeline(
    source_text: str,
    protected_facts_text: str,
    target_grade: int,
    max_refinements: int,
    exercise_refiner: bool,
):
    case = build_case_from_ui(source_text, protected_facts_text, target_grade)
    result = await run_pipeline(
        case,
        max_refinements=max_refinements,
        exercise_refiner=exercise_refiner,
    )

    acceptance = result["acceptance"]
    status = (
        "## HUMAN REVIEW REQUIRED\n"
        f"- Automated verifier and deterministic checks: "
        f"{'passed' if result['automated_checks_passed'] else 'issues found'}\n"
        f"- Refiner runs: {result['refinements']} / {result['max_refinements']}\n"
        "- Stage path: Simplifier → Readability → Verifier"
        + (
            " → Refiner → Verifier" * result["refinements"]
            if result["refinements"]
            else ""
        )
        + "\n"
        "- Deterministic findings require review but do not by themselves trigger "
        "Refiner retries.\n"
        "- This prototype never automatically approves clinical text."
    )
    return (
        status,
        result["draft"],
        result["readability_draft"],
        result["initial_candidate"],
        json.dumps(result["initial_verifier_feedback"], indent=2),
        json.dumps(result["verifier_feedback"], indent=2),
        result["refinements"],
        result["final_text"],
        json.dumps(acceptance, indent=2),
        json.dumps(result, indent=2),
    )


async def run_adversarial_ui():
    from agents.evaluate_verifier_cases import evaluate_cases

    return await evaluate_cases()


def create_demo() -> gr.Blocks:
    dataset_examples = load_dataset_examples()
    dataset_status = (
        f"Loaded {sum(label.startswith('MTSamples |') for label in dataset_examples)} "
        "long MTSamples notes and "
        f"{sum(label.startswith('MedAESQA |') for label in dataset_examples)} "
        "MedAESQA expert answers from the local data directory."
        if dataset_examples
        else "No supported local dataset files found. You can still enter synthetic text."
    )
    with gr.Blocks(title="FaithfulMed Multi-Agent Review") as demo:
        gr.Markdown(
            "# FaithfulMed Multi-Agent Prototype\n"
            "Run the Simplifier → Readability → Verifier → bounded Refiner flow. "
            f"Model: `{os.getenv('FAITHFULMED_MODEL', 'gemini-flash-latest')}`. "
            "Use synthetic or approved de-identified notes only. Output always "
            "requires human review."
        )
        gr.Markdown(
            "Dataset records are read from this machine. Loading a record only fills "
            "the source box; clicking Run sends that text to the configured model. "
            "MedAESQA expert answers are detailed QA answers, not clinical notes."
        )
        with gr.Row():
            with gr.Column(scale=2):
                dataset = gr.Dropdown(
                    choices=list(dataset_examples),
                    value=next(iter(dataset_examples), None),
                    label="Optional local long-form dataset example",
                    interactive=bool(dataset_examples),
                )
                dataset_info = gr.Markdown(dataset_status)
                load_dataset_button = gr.Button(
                    "Load selected example into source note",
                    interactive=bool(dataset_examples),
                )
                source = gr.Textbox(
                    label="Source note",
                    value=SYNTHETIC_SOURCE,
                    lines=7,
                )
                protected = gr.Textbox(
                    label="Critical facts to require exactly (one per line)",
                    value=DEFAULT_PROTECTED_FACTS,
                    lines=6,
                    info=(
                        "Each phrase must occur in the source; a rewrite that changes "
                        "it is flagged for human review."
                    ),
                )
                target = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=8,
                    step=1,
                    label="Target reading grade",
                )
                max_refinements = gr.Slider(
                    minimum=0,
                    maximum=MAX_REFINEMENTS,
                    value=MAX_REFINEMENTS,
                    step=1,
                    label="Maximum Refiner calls",
                )
                exercise = gr.Checkbox(
                    label="Inject false 'confirmed' claim to test Verifier/Refiner",
                    value=False,
                )
                run_button = gr.Button("Run multi-agent pipeline", variant="primary")
                adversarial_button = gr.Button("Run live Verifier adversarial suite")
                load_dataset_button.click(
                    fn=lambda selection: load_selected_dataset_case(
                        selection,
                        dataset_examples,
                    ),
                    inputs=[dataset],
                    outputs=[source, protected],
                )
            with gr.Column(scale=3):
                status = gr.Markdown()
                with gr.Tab("Stages"):
                    draft = gr.Textbox(label="Simplifier draft", lines=5)
                    readable = gr.Textbox(
                        label="Readability output",
                        lines=5,
                    )
                    injected = gr.Textbox(
                        label="Candidate sent to Verifier (normally the Readability output)",
                        lines=5,
                    )
                    final_text = gr.Textbox(
                        label="Final candidate (after zero or more Refiner calls)",
                        lines=6,
                    )
                with gr.Tab("Verifier and review"):
                    initial_feedback = gr.Code(
                        label="Initial Verifier feedback",
                        language="json",
                    )
                    final_feedback = gr.Code(
                        label="Final Verifier feedback",
                        language="json",
                    )
                    refinement_count = gr.Number(label="Refiner calls", precision=0)
                    acceptance = gr.Code(
                        label="Deterministic checks and human-review status",
                        language="json",
                    )
                with gr.Tab("Full run data"):
                    full_result = gr.Code(label="Pipeline JSON", language="json")
                with gr.Tab("Adversarial suite"):
                    adversarial_result = gr.JSON(
                        label="Live Verifier case results",
                    )

        run_button.click(
            fn=run_ui_pipeline,
            inputs=[source, protected, target, max_refinements, exercise],
            outputs=[
                status,
                draft,
                readable,
                injected,
                initial_feedback,
                final_feedback,
                refinement_count,
                final_text,
                acceptance,
                full_result,
            ],
        )
        adversarial_button.click(
            fn=run_adversarial_ui,
            inputs=[],
            outputs=[adversarial_result],
        )
        gr.Markdown(
            "The live adversarial suite sends seven synthetic cases to the Verifier "
            "and may consume API quota. Deterministic checks are conservative and "
            "cannot prove semantic equivalence."
        )
    return demo


if __name__ == "__main__":
    create_demo().launch(server_name="127.0.0.1", inbrowser=True)
