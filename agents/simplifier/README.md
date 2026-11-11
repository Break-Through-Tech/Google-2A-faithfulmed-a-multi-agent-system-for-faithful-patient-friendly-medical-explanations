# FaithfulMed Simplifier

The Simplifier creates a patient-friendly draft from source text and extracted
clinical atoms. It is called directly through the project's shared Gemini model
runner. Optional `retrieved_context` may provide general definitions for
terminology; it is never evidence about the patient.

## Shared input contract

`PipelineCase` validates the Extractor's `atoms` list and verifies that each
`source_quote` appears in the source note. It assigns stable atom IDs and adapts
the case for downstream agents.

```python
from agents.contracts import PipelineCase

case = PipelineCase.from_extractor_output(
    source_text=source_note,
    extractor_output=extractor_result,
    target_grade=8,
)
state = case.to_state()
```

The interactive sequence is Extractor → Simplifier → Readability → Verifier.
When the Verifier finds issues, Refiner produces a candidate that is sent back
to Verifier. The Gradio interface lets you run each agent separately and see
the exact input/output at every handoff.

## Run the UI and synthetic pipeline

From the repository root, install requirements and securely set `GEMINI_API_KEY`
in PowerShell, then launch either the step-by-step UI or the command-line
Simplifier → Readability → Verifier pipeline:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$secureKey = Read-Host -AsSecureString "Gemini API key"
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
.\.venv\Scripts\python.exe -m agents.ui
```

```powershell
.\.venv\Scripts\python.exe -m agents.run_pipeline
.\.venv\Scripts\python.exe -m agents.run_pipeline --exercise-refiner
```

The command-line workflow uses a synthetic cough note and runs up to two
Verifier/Refiner rounds. `--exercise-refiner` adds a deliberately unsupported
certainty claim to test the verification path. It makes live Gemini calls and
may use API quota. The tests inject fake executors and run without a key:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_handoffs tests.test_agent_workflow -v
```

## Optional MedlinePlus retrieval

The local MedlinePlus index is not committed. The retrieval helper and
Simplifier implementation can use the index built by
`agents/medlineplus_rag_index.ipynb`. Retrieved definitions may help choose
plain-language wording but must not add patient-specific information or advice.
Use synthetic or approved de-identified notes only.
