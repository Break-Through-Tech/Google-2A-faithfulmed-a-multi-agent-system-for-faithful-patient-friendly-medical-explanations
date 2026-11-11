import csv
import json
import tempfile
import unittest
from pathlib import Path

from agents.ui import (
    build_case_from_ui,
    create_demo,
    load_dataset_examples,
    load_selected_dataset_case,
)


class UserInterfaceTests(unittest.TestCase):
    def test_builds_pipeline_case_from_note_and_manual_facts(self):
        case = build_case_from_ui(
            "The patient reports a cough for three days.",
            "cough for three days\nreports a cough",
            7,
        )

        self.assertEqual(case.target_grade, 7)
        self.assertEqual(len(case.atoms), 2)
        self.assertEqual(case.protected_facts[0], "cough for three days")

    def test_rejects_protected_fact_not_in_source(self):
        with self.assertRaisesRegex(ValueError, "source_quote"):
            build_case_from_ui(
                "The patient reports a cough.",
                "has pneumonia",
                8,
            )

    def test_loads_long_local_dataset_examples_into_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            with (data_dir / "mtsamples_baseline_outputs.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["sample_name", "transcription"],
                )
                writer.writeheader()
                writer.writerow({
                    "sample_name": "Long note",
                    "transcription": "Clinical note. " * 250,
                })
            (data_dir / "medaesqa_v1.json").write_text(
                json.dumps([
                    {
                        "question_id": "test-1",
                        "question": "What is the condition?",
                        "expert_curated_answer": "Detailed expert answer. " * 100,
                    }
                ]),
                encoding="utf-8",
            )

            examples = load_dataset_examples(data_dir, limit_per_dataset=1)

        self.assertEqual(len(examples), 2)
        mtsamples_label = next(
            label for label in examples if label.startswith("MTSamples |")
        )
        self.assertIn("words", mtsamples_label)
        source_text, protected_facts = load_selected_dataset_case(
            mtsamples_label,
            examples,
        )
        self.assertGreater(len(source_text), 3000)
        self.assertEqual(protected_facts, "")

    def test_rejects_unknown_dataset_selection(self):
        with self.assertRaisesRegex(ValueError, "available local dataset"):
            load_selected_dataset_case("unknown sample", {})

    def test_creates_gradio_app(self):
        app = create_demo()

        try:
            self.assertGreater(len(app.get_config_file()["components"]), 0)
        finally:
            app.close(verbose=False)


if __name__ == "__main__":
    unittest.main()
