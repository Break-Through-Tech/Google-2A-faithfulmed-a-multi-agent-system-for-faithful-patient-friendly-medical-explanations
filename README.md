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

### Agent pipeline and local UI

The agents call Gemini directly through `google-genai`; orchestration and
handoffs are managed by the project code rather than an agent framework. The
shared `PipelineCase` validates Extractor atoms and adapts them for the
Simplifier, Readability, Verifier, and Refiner. See the [Simplifier README](agents/simplifier/README.md)
and [Readability README](agents/readability/README.md) for details.

Install dependencies and run the local workbench from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$secureKey = Read-Host -AsSecureString "Gemini API key"
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
$env:FAITHFULMED_MODEL = "gemini-3.5-flash-lite"
.\.venv\Scripts\python.exe -m agents.ui
```

Open the localhost URL printed by Gradio (normally `http://127.0.0.1:7860`).
Use **Run next in sequence** to proceed through Extractor → Simplifier →
Readability → Verifier, with Refiner inserted when Verifier reports issues.
Or select and run an individual agent. Each result shows that agent's exact
input, output, and next handoff. After Refiner, run Verifier again; refinement
is capped at two runs. Reset starts a fresh sequence. Changing the note or
settings requires rerunning Extractor to prevent stale inputs.

Dataset examples are read from this machine and are not sent until an agent is
run. If MedlinePlus retrieval is enabled, its query is also sent to Gemini for
embedding. Use synthetic or approved de-identified notes only, and keep the
interface bound to localhost. Every run requires human review; a passing
Verifier is not proof of clinical correctness. The optional MedlinePlus index is
local and not committed; retrieved definitions may help explain terminology
but never establish patient-specific facts.

The Readability output is measured independently using Flesch-Kincaid grade
(the primary <= 8 target), Flesch Reading Ease, SMOG, Coleman-Liau, Automated
Readability Index, Gunning Fog, Dale-Chall, sentence length, and difficult-word
ratio. These are complementary estimates, not guarantees; difficult-word ratio
is not a medical-jargon detector. `assess_candidate` checks protected facts and
keeps a failed rewrite from silently replacing its input.

Run the synthetic pipeline or offline tests:

```powershell
.\.venv\Scripts\python.exe -m agents.run_pipeline
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The pipeline and handoff tests use injected fake executors and need no Gemini
key. `python -m agents.evaluate_verifier_cases` runs live synthetic cases and
uses Gemini API quota.

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
