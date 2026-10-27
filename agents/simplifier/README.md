# FaithfulMed Simplifier Agent

This is a minimal standalone Google ADK Simplifier. It produces the `draft`
state value from `source_text`, `atoms`, and `target_grade`. Optional
`retrieved_context` contains sourced general definitions; it is not required to
run the agent.

## Shared input contract

Build a validated `PipelineCase` from the source note and Extractor notebook
output:

```python
from agents.contracts import PipelineCase

case = PipelineCase.from_extractor_output(
    source_text=source_note,
    extractor_output=extractor_result,
    target_grade=8,
)
adk_state = case.to_adk_state()
```

The canonical atom fields are `id`, `text`, `type`, and `source_quote`.
`from_extractor_output` validates the Extractor's `atoms` list and verifies
that each exact source quote occurs in the source note. It assigns stable
`atom-001`, `atom-002`, ... IDs for the case. The adapter
`case.verifier_atoms()` maps atoms to the existing Verifier fields
(`id`, `concept`, and `importance`). `case.refiner_state(candidate_text,
verifier_feedback)` maps source and candidate text to the Refiner's existing
`original_text`, `simplified_text`, and `verifier_feedback` inputs.

The expected first-pass handoff is:

```text
Extractor result -> PipelineCase -> Simplifier (`draft`)
                 -> Readability (`readability_draft`)
                 -> Verifier (`verifier_feedback`)
                 -> Refiner (`simplified_text`) -> Verifier
```

The pipeline should pass the Readability result as the Verifier's
`simplified_text` and the Refiner's candidate. Re-run the Verifier after each
refinement. `agents.run_pipeline` now implements those handoffs with individual
in-memory ADK runners and a configurable bounded retry limit.

## Run the synthetic end-to-end handoff

From the repository root, set a valid Gemini API key in PowerShell, then run:

```powershell
Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue
$env:GEMINI_API_KEY = $null
$secureKey = Read-Host -AsSecureString "Gemini API key"
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
.\.venv\Scripts\python.exe -m agents.run_pipeline
```

The script uses a synthetic cough note and Extractor-shaped sample output. It
runs Simplifier, Readability, Verifier, and up to two Refiner/re-verification
rounds, then prints a JSON result with each candidate and the final verifier
status. It prints a progress line before and after each Gemini call; a long
pause after a "Calling Gemini" line means that agent is waiting on the API.
Use Ctrl+C to cancel if it does not return. Use `--max-refinements 0` to test
the no-refinement path. This makes live Gemini calls; 503/429 responses may be
temporary and incur API usage.

To deliberately test whether Verifier catches a changed certainty and whether
Refiner corrects it, run:

```powershell
.\.venv\Scripts\python.exe -m agents.run_pipeline --exercise-refiner
```

This appends the false claim "The suspected viral infection is confirmed." to
the Readability output before verification. Review `initial_candidate`,
`initial_verifier_feedback`, `refiner_exercised`, `final_text`, and the final
`verifier_feedback` in the JSON output. A successful demonstration means the
initial verifier flags the claim, Refiner is called, and final verification is
run again. If the verifier passes the injected false claim, the demonstration
has exposed a verifier miss; it is not a successful faithfulness result.

The runner module supports an injected async agent executor so the state
handoffs and retry cap can be tested offline:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_handoffs
```

## Retrieval

No vector index is needed to test the first multi-agent handoffs. Leave
`retrieved_context` empty and verify source-grounded rewriting, schema
validation, and the Verifier/Refiner cycle first. Add RAG when the team has
approved source documents, a repeatable indexing process, and tests for source
metadata and retrieval quality. Retrieved definitions may help explain terms,
but the source note remains authoritative for patient-specific details.
