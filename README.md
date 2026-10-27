# FaithfulMed: A Multi-Agent System for Faithful, Patient-Friendly Medical Explanations

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Aanchal Puri     | @aanchalpuri  | Developed the vector retrieval index using Gemini embeddings and ChromaDB (RAG pipeline) over the MedlinePlus glossary and health guidelines  |
| Arshiya Salehi   | @arshiya-salehi | .... |
| Rudra Patel      | @rudra-045    | Added refusal detection and aggregate scoring to the shared evaluation harness; integrated and evaluated the 50 single-agent Gemini baseline outputs. |
| Jia Panjwani     | @jiap6        | .... |
| Aliaa Mahgoub    | @AliaaMahgoub | Performed MedlinePlus exploratory data analysis, contributed to vector retrieval index with Aanchal |
| Shayna Jain      | @shaynajain   | Created medaesqa_eda.ipynb; developed mtsamples_baseline.ipynb, including dataset curation and single-agent Gemini baseline generation. |
| Lauren Cullari   | @lcullari     | Did an exploratory data analysis on MedAESQA, curated the ~50 sample dataset for the single-agent baseline using MTSamples with Shayna. |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

### Readability agent

The Readability agent is a standalone Google ADK agent in `agents/readability/`.
See the [Readability Agent README](agents/readability/README.md) for complete
PowerShell setup, run, evaluation, and multi-agent integration instructions.
Its output is measured independently of the model using Flesch-Kincaid grade level
(the project's primary <= 8 target), Flesch Reading Ease, SMOG, Coleman-Liau,
Automated Readability Index, Gunning Fog, Dale-Chall, sentence length, and a
difficult-word ratio. The difficult-word ratio is a general vocabulary proxy, not
a medical-jargon detector. These formulas are complementary signals, not
interchangeable grade-level guarantees.

Run its ADK development UI from the repository root:

```powershell
Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue
$env:GEMINI_API_KEY = $null
$secureKey = Read-Host -AsSecureString "Gemini API key"
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
adk web agents
```

Set the key in the same PowerShell session used to start ADK. The Google Gen AI
SDK prefers `GOOGLE_API_KEY` if both it and `GEMINI_API_KEY` are set, so remove
any stale `GOOGLE_API_KEY` first. Restart ADK after changing keys. If a key was
shared in chat or logs, revoke it in AI Studio and use a replacement.

The evaluator can score the checked-in MTSamples baseline or another CSV with a
text column:

```powershell
python -m agents.readability.evaluate data/mtsamples_baseline_outputs.csv
python -m unittest discover -s tests
```

On the currently checked-in 50-output baseline, this evaluator reports 72% at or
below grade 8 by Flesch-Kincaid (mean grade 7.65). Mean SMOG is 10.48; the other
readability measures are included in the evaluator output. These are baseline
measurements, not evidence that the Readability agent has improved the outputs.

`assess_candidate` checks caller-supplied protected clinical facts after generation.
If any are missing, it returns the original draft and marks the result
`preservation_failed`; it does not silently accept the rewrite. Use short,
specific protected facts such as medication names, doses, lab values, dates, and
instructions. The final Verifier should still validate the accepted readability
output.

The Readability agent accepts optional `glossary_context` from the upstream
Simplifier/RAG step; it does not create a second retrieval index. To export ADK
spans and structured telemetry to Google Cloud Trace and Cloud Logging, set
`FAITHFULMED_CLOUD_OBSERVABILITY=true`, configure Google Cloud Application Default
Credentials with permissions for the selected project, then start `adk web agents`.
Telemetry does not enable prompt-content capture. Do not enable content capture for
clinical text unless the data handling policy explicitly permits it.

References: [ADK LLM agents](https://adk.dev/agents/llm-agents/),
[ADK observability](https://adk.dev/observability/), and
[ADK traces](https://adk.dev/observability/traces/).

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
