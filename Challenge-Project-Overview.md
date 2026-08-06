---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟢 | Stack relies on the standard Python ecosystem and Hugging Face libraries; it is perfectly compatible with the Google Colab environment. |
| Data Readiness | 🟡 | Datasets are standard but require significant orchestration and standardization from the FHIR format before integration into an RAG pipeline. |
| Resource Check | 🟡 | Reliance on the Hugging Face Inference API can hit rate limits; local model execution on the Colab free tier is limited by memory constraints for multi-agent loops. |

### Internal Scores
- **Student Fit Score:** 6.5/10
- **Technical Depth Score:** 8.5/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
The project addresses a critical industry need with a sophisticated architectural approach. To ensure success, prioritize: 1) Stabilizing the agent logic by replacing recursive agent loops with a deterministic Directed Acyclic Graph (DAG) flow. 2) Establishing a static 'Gold Standard' evaluation set early to prevent model drift during iteration. Please refine the scope to ensure delivery within the 12-week cap.

---

# FaithfulMed: A Multi-Agent System for Faithful, Patient-Friendly Medical Explanations

**Company / Org:** Google  
**Challenge Advisor:** Sarita Anand Joshi, sarita.ritu@gmail.com  
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Google
Google is a global technology leader focused on organizing the world's information and making it universally accessible and useful. 

---

## 🎯 The Challenge
### Project Summary
In this project, you will use publicly available de-identified clinical text (MTSamples transcribed medical reports, Synthea synthetic patient records, MedQuAD consumer health Q&A, and the PLABA plain-language adaptation corpus) and a multi-agent LLM architecture combining retrieval-augmented generation, prompt engineering, source-grounded faithfulness verification, and iterative agent-based refinement using open-source models (Llama-3, Mistral, Gemma) to build a system in which specialized agents (Extractor, Simplifier, Verifier, Refiner, Readability) collaborate to produce patient-friendly explanations of clinical documents that are simultaneously easy to read and provably faithful to the source. This will help our organization address the well-documented health-literacy gap — where patients routinely misunderstand discharge instructions, lab results, and care plans written above their reading level — while tackling the central blocker to deploying LLMs for patient-facing use: hallucinated or unfaithful information that could harm patients if trusted.

### Success Criteria
A successful December outcome includes:

_Quantitative system performance_
- Readability: ≥80% of outputs at ≤8th-grade reading level (Flesch-Kincaid).
- Faithfulness: ≥85% factual fidelity on the human-annotated test set, verified against human labels.
- Hallucination rate: <10% of outputs contain a clinically meaningful unsupported claim.
- Multi-agent vs. single-agent baseline: ≥15% absolute improvement in faithfulness score.

_Verifier agent quality_
- Verifier agrees with human annotators ≥80% of the time on faithfulness labels.

_Ablation analysis (research contribution)_
- A complete ablation table showing the marginal contribution of each agent — answers the question "which agents matter most for faithfulness vs. readability?"

_Comparative analysis_
- Published leaderboard across ≥3 open models showing faithfulness, readability, latency, and cost tradeoffs.

_Working artifacts_
- Public GitHub repo with reproducible code and one notebook per agent.
- Hosted demo on Hugging Face Spaces showing per-agent traces.
- Technical report (~10 pages) including the ablation study, final team presentation.
- Portfolio-ready writeup
- Each fellow can speak to their owned agent in interviews and on LinkedIn ("I designed and validated the Faithfulness Verifier agent against human-annotated medical text").

### Stretch Goals

For teams that progress quickly:
- Multilingual extension — add Spanish generation with a parallel Verifier (biggest equity impact; Spanish-speaking US populations face larger health-literacy gaps).   
- Personalization layer — adjustable target reading level and tone (formal / conversational) through the Readability agent.   
- Agent debate variant — replace single Verifier with a two-agent debate  compare against single-Verifier ablation.   
- Bias & fairness audit — does the pipeline perform worse on reports involving non-English names, rare conditions, or specific demographic markers? Run a structured audit and publish findings.   
- LoRA fine-tuning experiment — fine-tune a small open model (e.g., Gemma-2-2B) on patient-friendly explanation pairs and compare against prompt-only baselines.   
- Domain expansion — apply the pipeline to a high-impact subdomain (oncology discharge summaries, post-partum care instructions, pediatric medication labels).   
- Tool-augmented agents — give the Extractor a UMLS lookup tool and the Verifier a DrugBank lookup tool; measure whether tools improve verification accuracy.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Foundations & Single-Agent Baseline | • Onboard the team: GitHub repo, Colab environment, and free-tier API access (Groq for Llama-3, Hugging Face Inference API).<br>• Exploratory data analysis on MTSamples, MedQuAD, PLABA, and a Synthea-generated synthetic-patient sample.<br>• Define the task spec, success rubric, and shared evaluation harness: Flesch-Kincaid, SMOG, medical-jargon density, output length, refusal rate.<br>• Build a single-LLM baseline (no agents) on ~50 hand-curated examples — this becomes the comparison point for the multi-agent system.<br>• Each fellow chooses one of the five agents to own end-to-end through the project. |
| October | Multi-Agent Pipeline & Verifier Calibration | • Implement Extractor Agent: structured-output prompting that returns a list of clinical atoms with types.<br>• Implement Simplifier Agent with RAG over MedlinePlus lay-language glossary.<br>• Implement Verifier Agent: LLM-as-judge module inspired by FaithJudge, scoring faithfulness, omission, addition, and reading-level match.<br>• Create a 100-example human-annotated test set across the team for Verifier calibration; target ≥80% agreement with human labels (Cohen's κ ≥ 0.6).<br>• Wire up the Refiner and Readability agents with bounded iteration limits.<br>• Run the first end-to-end multi-agent vs. baseline comparison across 3+ open models. |
| November | Ablations, Demo & Deliverables | • Ablation studies: drop one agent at a time and measure impact on each metric — this is the research contribution.<br>• Error analysis: which clinical content types (lab values, drug names, procedures) produce the most hallucinations? Which agent catches them?<br>• Build an App demo on Hugging Face Spaces (free) showing per-agent traces, faithfulness scores, and flagged passages.<br>• Final deliverables: GitHub repo with reproducible Colab notebooks, technical report, FaithfulMed leaderboard, ablation table, and team presentation.<br>• Optional : arXiv paper draft |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** [TBD]  
**Format:** Text, FHIR (JSON-based clinical standards)  
**Size:** under 1gb  
**Location:** https://www.mtsamples.com/, https://synthetichealth.github.io/synthea/

### Key Details
- [Brief description of what's in the data]
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]
  
---

## 🛠️ Suggested Approach

**ML Problem Type:** [e.g., Classification, Regression, NLP, Computer Vision, LLM/RAG]

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]
  
---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?
Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session C).
