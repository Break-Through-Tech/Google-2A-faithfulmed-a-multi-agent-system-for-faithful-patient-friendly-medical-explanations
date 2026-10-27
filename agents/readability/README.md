# FaithfulMed Readability Agent

This folder contains the standalone Google ADK Readability agent. It rewrites a
patient-facing medical explanation in clear language, aiming for a
Flesch-Kincaid grade level of 8 or below while preserving the clinical meaning.

The agent does not build or call the retrieval index itself. In the eventual
pipeline, the Simplifier can pass relevant retrieved definitions as
`retrieved_context`.

If ADK Web is already running while you change Python files, stop and restart
the server so it reloads the agent package.

## Requirements

- Python 3.10 or later
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Dependencies from the repository's `requirements.txt`

Run the following commands in PowerShell from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists, skip its creation. Stop any running ADK server with
Ctrl+C. In the same PowerShell window that you will use to start ADK, remove a
stale `GOOGLE_API_KEY` (the Google Gen AI SDK prefers it over
`GEMINI_API_KEY`) and set your current Gemini key without placing it in shell
history:

```powershell
Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue
$env:GEMINI_API_KEY = $null
$secureKey = Read-Host -AsSecureString "Gemini API key"
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
```

Then start ADK in that same window. The key is only set for that PowerShell
session. Never commit a key or paste it into a notebook. The repository ignores
`.env` files if you prefer ADK's local-file configuration; keep the two API-key
variables from conflicting if you use that approach.

If Google responds with `API_KEY_INVALID`, verify the key in AI Studio, replace
it if it was revoked or exposed, confirm the `GOOGLE_API_KEY` variable is
removed, and restart ADK from the PowerShell session where `GEMINI_API_KEY` is
set.

To use a different supported model, set `FAITHFULMED_MODEL` before starting ADK.
The default is `gemini-flash-latest`.

## Try the agent with ADK Web

From the repository root, start the local development UI:

```powershell
.\.venv\Scripts\adk.exe web agents
```

Open the local URL printed by ADK (normally `http://127.0.0.1:8000`) and select
`readability`. ADK Web is a development tool; keep it on a trusted local
machine/network and do not expose its unauthenticated server publicly. If ADK
asks about optional CLI usage telemetry on first run, choose according to your
preference; this is separate from FaithfulMed Cloud observability.

Try a synthetic example such as:

```text
Rewrite this note for a patient. Keep all facts and uncertainty unchanged.
The patient reports a dry cough for three days. No fever or shortness of breath
is reported. A viral upper respiratory infection is suspected. The plan is rest,
fluids, and follow-up if symptoms worsen or persist.
```

Check the answer manually for the dry cough and its duration, the absence of
fever and shortness of breath, the uncertainty ("suspected"), and the plan.
Compare readability scores with the original using the evaluator below. A model
response is not automatically clinically verified just because it reads more
easily.

## Run automated checks

Run the unit tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover metric calculation, invalid input, the protected-fact fallback,
and privacy-safe metric logging. They do not make a Gemini API call.

### Score the existing baseline

The CSV evaluator scores text without making a model call:

```powershell
.\.venv\Scripts\python.exe -m agents.readability.evaluate `
  data\mtsamples_baseline_outputs.csv `
  --text-column simplified_transcription
```

It reports the number of outputs, share at or below grade 8, and mean values
for:

- Flesch-Kincaid grade level (the project's primary target measure)
- Flesch Reading Ease
- SMOG
- Coleman-Liau
- Automated Readability Index (ARI)
- Gunning Fog
- Dale-Chall
- Sentence length and difficult-word ratio

These formulas are complementary estimates, not interchangeable scores.
`difficult_word_ratio` uses a general familiar-word list; it is **not** a
medical-jargon detector. The current 50-output baseline scored 72% at or below
grade 8 by Flesch-Kincaid, with a mean grade of 7.65. This is the baseline, not
an evaluation of the Readability agent.

To measure a new set of agent outputs, save them in a CSV column and pass that
column name with `--text-column`. Keep a stable sample set and compare each
agent output with the matching original draft.

## Use the post-generation preservation check

`assess_candidate` measures a generated rewrite and checks caller-provided facts
by normalized exact phrase matching:

```python
from agents.readability.metrics import assess_candidate

result = assess_candidate(
    original_draft=original_text,
    candidate=readability_agent_text,
    protected_facts=[
        "metformin",
        "500 mg",
        "twice daily",
        "follow up in two weeks",
    ],
)

print(result.status)
print(result.candidate_metrics.flesch_kincaid_grade)
print(result.missing_protected_facts)
```

Possible statuses are:

- `accepted`: protected phrases are present and the Flesch-Kincaid target is met.
- `target_not_met`: protected phrases are present, but the grade target is not met.
- `preservation_failed`: one or more protected phrases are missing; `output_text`
  falls back to the original draft.

The check is deliberately conservative and is **not** a semantic clinical
faithfulness verifier. Supply concise, exact protected values and instructions.
Integrators must call `assess_candidate` after the ADK agent and must not return
the candidate as accepted when preservation fails. Run the Verifier after
Readability because Readability changes the wording.

## Integration contract for the other agents

The agent's ADK `output_key` is `readability_draft`.

| State key | Direction | Meaning |
|---|---|---|
| `draft` | Input | Patient-facing draft produced by the Simplifier. Optional only for standalone chat; provide it in a pipeline. |
| `target_grade` | Input | Optional numeric reading-level target; defaults to 8 in the agent instructions. |
| `atoms` | Input | Optional structured clinical facts from the Extractor; the agent is instructed to preserve them. |
| `protected_facts` | Input | Optional exact phrases for high-risk details that the deterministic check should preserve. |
| `retrieved_context` | Input | Optional sourced definitions retrieved upstream; use for wording, not patient-specific claims. |
| `readability_draft` | Output | The agent's revised text. |

The future pipeline should:

1. Produce the draft, facts, protected phrases, and optional glossary context.
2. Run the Readability agent and read its `readability_draft` state output.
3. Call `assess_candidate` against the original draft and protected phrases.
4. Route a `preservation_failed` result to the Refiner or an explicit
   `review_required` outcome. Do not silently fall back and present it as
   successfully simplified.
5. Run the Verifier on the final candidate after Readability and any refinement.
6. Store per-agent output and metric results for evaluation, using synthetic or
   approved de-identified examples.

The existing `notebooks\adk_pipeline.py` is an integration scaffold. It includes
the Readability agent before the Verifier/Refiner loop and a final verification
step. The deterministic `assess_candidate` check is not automatically called by
the scaffold yet; wire it into the application before treating candidate output
as accepted.

The current terminal pipeline and local UI additionally call
`agents.acceptance.assess_acceptance`, which checks caller-supplied critical
phrases, numeric/timing details, negation, uncertainty, and reported status.
Those conservative checks can flag safe paraphrases and cannot prove semantic
faithfulness. The UI and pipeline always return `human_review_required`, even
when the LLM Verifier passes.

For the full local multi-agent UI, including stage outputs, verifier findings,
Refiner count, deterministic checks, and review status, run
`.\.venv\Scripts\python.exe -m agents.ui` from the repository root. See the
project README's **Local multi-agent review UI** section for setup.

## Optional Cloud Trace and Cloud Logging

Cloud export is disabled by default. To opt in, configure Google Cloud
Application Default Credentials, select a project with the required APIs and
permissions, and enable the flag before importing the agent:

```powershell
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
$env:FAITHFULMED_CLOUD_OBSERVABILITY = "true"
.\.venv\Scripts\adk.exe web agents
```

Cloud usage may incur charges. The agent records spans and structured readability
metrics; its custom metrics log does not include drafts or generated text.
ADK can capture prompt/response content if explicitly configured to do so; do
not enable content capture for clinical text unless your data policy permits it.
Clear the opt-in flag when you are done:

```powershell
Remove-Item Env:FAITHFULMED_CLOUD_OBSERVABILITY
```

References: [ADK Python quickstart](https://adk.dev/get-started/python/),
[ADK LLM agents](https://adk.dev/agents/llm-agents/),
[ADK observability](https://adk.dev/observability/), and
[ADK traces](https://adk.dev/observability/traces/).
