# FaithfulMed Readability Agent

The Readability agent rewrites a patient-facing draft in clear language while
preserving supported clinical information. It is called directly through the
shared `google-genai` model runner and is exposed as an individual step in the
local Gradio workbench.

## Run the local workbench

From the repository root, install the project requirements and provide a Gemini
API key in the same PowerShell session:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$secureKey = Read-Host -AsSecureString "Gemini API key"
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
.\.venv\Scripts\python.exe -m agents.ui
```

In the UI, run Extractor, Simplifier, and then Readability. The readability
step displays the draft it received, the generated rewrite, and deterministic
readability and protected-fact results. The Verifier should review the final
candidate after Readability and any Refiner pass.

Use synthetic or approved de-identified notes only. The app binds to localhost
by default. A generated rewrite is never automatically approved.

## Evaluation

Run the unit tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_readability tests.test_agent_workflow -v
```

Score the checked-in MTSamples baseline without making a model call:

```powershell
.\.venv\Scripts\python.exe -m agents.readability.evaluate `
  data\mtsamples_baseline_outputs.csv `
  --text-column simplified_transcription
```

The evaluator reports Flesch-Kincaid grade level (the primary <= 8 target),
Flesch Reading Ease, SMOG, Coleman-Liau, Automated Readability Index, Gunning
Fog, Dale-Chall, sentence length, and difficult-word ratio. These are
complementary estimates, not interchangeable scores. The difficult-word ratio
is a general vocabulary proxy, not a medical-jargon detector.

`assess_candidate` checks caller-provided protected phrases after generation.
If a phrase is missing, it returns the original draft and marks the result
`preservation_failed`; this deterministic check is not a semantic faithfulness
guarantee. The Verifier remains a separate step and every output requires human
review.
