"""Local Gradio interface for stepping through agents and inspecting handoffs."""

import csv
import json
from pathlib import Path
from typing import Mapping

import gradio as gr

from agents.contracts import PipelineCase
from agents.run_pipeline import SYNTHETIC_SOURCE
from agents.step_runner import STAGES, run_agent_stage


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
    """Build a shared case from user-entered source and optional exact facts."""
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
        ],
        "not_found_types": [],
    }
    return PipelineCase.from_extractor_output(
        source_text=source_text,
        extractor_output=extractor_output,
        protected_facts=facts,
        target_grade=target_grade,
    )


async def run_ui_stage(
    source_text: str,
    protected_facts_text: str,
    target_grade: int,
    use_retrieval: bool,
    state: dict | None,
    stage: str,
):
    protected_facts = [
        line.strip()
        for line in protected_facts_text.splitlines()
        if line.strip()
    ]
    updated_state, result = await run_agent_stage(
        stage,
        source_text,
        protected_facts,
        int(target_grade),
        state,
        use_retrieval=use_retrieval,
    )
    next_stage = updated_state.get("next_stage")
    status = (
        f"Completed **{stage}**. Next in sequence: **{next_stage}**."
        if next_stage
        else f"Completed **{stage}**. The sequence is complete; review every output."
    )
    return updated_state, status, result, updated_state["history"]


async def run_next_ui_stage(
    source_text: str,
    protected_facts_text: str,
    target_grade: int,
    use_retrieval: bool,
    state: dict | None,
):
    next_stage = (state or {}).get("next_stage", "Extractor")
    if next_stage is None:
        raise ValueError("No next agent is queued. Review results or reset the run.")
    return await run_ui_stage(
        source_text,
        protected_facts_text,
        target_grade,
        use_retrieval,
        state,
        next_stage,
    )


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

    with gr.Blocks(title="FaithfulMed Agent Workbench") as demo:
        gr.Markdown(
            "# FaithfulMed Agent Workbench\n"
            "Run one agent at a time, inspect its exact input and output, then pass "
            "the result to the next agent. The default order is "
            "**Extractor → Simplifier → Readability → Verifier → Refiner**; after "
            "Refiner, the candidate returns to Verifier. Outputs require human review."
        )
        gr.Markdown(
            "Use synthetic or approved de-identified notes only. Clicking an agent "
            "sends its input to the configured Gemini model. Dataset records are "
            "loaded locally and are not sent until an agent is run."
        )
        run_state = gr.State({})
        with gr.Row():
            with gr.Column(scale=2):
                dataset = gr.Dropdown(
                    choices=list(dataset_examples),
                    value=next(iter(dataset_examples), None),
                    label="Optional local dataset example",
                    interactive=bool(dataset_examples),
                )
                gr.Markdown(dataset_status)
                load_dataset_button = gr.Button(
                    "Load selected example",
                    interactive=bool(dataset_examples),
                )
                source = gr.Textbox(
                    label="Source note",
                    value=SYNTHETIC_SOURCE,
                    lines=8,
                )
                protected = gr.Textbox(
                    label="Critical facts to require exactly (one per line)",
                    value=DEFAULT_PROTECTED_FACTS,
                    lines=6,
                )
                target = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=8,
                    step=1,
                    label="Target reading grade",
                )
                use_retrieval = gr.Checkbox(
                    label="Use local MedlinePlus retrieval during Simplifier",
                    value=False,
                )
                selected_stage = gr.Dropdown(
                    choices=list(STAGES),
                    value=STAGES[0],
                    label="Agent to run",
                )
                with gr.Row():
                    run_selected = gr.Button(
                        "Run selected agent",
                        variant="primary",
                    )
                    run_next = gr.Button("Run next in sequence")
                    reset = gr.Button("Reset run")
            with gr.Column(scale=3):
                status = gr.Markdown("Ready. Start with the Extractor.")
                latest_result = gr.JSON(label="Latest agent input and output")
                all_results = gr.JSON(label="All completed agent steps")

        load_dataset_button.click(
            fn=lambda selection: load_selected_dataset_case(
                selection,
                dataset_examples,
            ),
            inputs=[dataset],
            outputs=[source, protected],
        )
        run_selected.click(
            fn=run_ui_stage,
            inputs=[
                source,
                protected,
                target,
                use_retrieval,
                run_state,
                selected_stage,
            ],
            outputs=[run_state, status, latest_result, all_results],
        )
        run_next.click(
            fn=run_next_ui_stage,
            inputs=[source, protected, target, use_retrieval, run_state],
            outputs=[run_state, status, latest_result, all_results],
        )
        reset.click(
            fn=lambda: ({}, "Ready. Start with the Extractor.", None, []),
            inputs=[],
            outputs=[run_state, status, latest_result, all_results],
        )
        gr.Markdown(
            "A rerun starts that stage again and invalidates dependent outputs. "
            "The Verifier can route to Refiner when it finds issues; after Refiner, "
            "run Verifier again. The UI never automatically approves clinical text."
        )
    return demo


if __name__ == "__main__":
    create_demo().launch(server_name="127.0.0.1", inbrowser=True)
