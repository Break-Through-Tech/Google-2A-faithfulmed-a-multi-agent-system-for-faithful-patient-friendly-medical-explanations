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
Google is a global technology leader focused on organizing the world's information and making it universally accessible and useful. This project aligns with Google’s commitment to advancing healthcare AI, specifically by aiming to bridge the critical health literacy gap through machine learning that makes clinical data more understandable for patients.

---

## 🎯 The Challenge
### Project Summary
This project involves building a multi-agent LLM architecture that transforms complex, clinical-grade medical documents into patient-friendly explanations. By coordinating specialized agents for extraction, simplification, and verification, the system utilizes RAG and open-source models to ensure outputs are both easy to read and clinically accurate, directly addressing the risks of hallucination in patient-facing AI.

### Success Criteria
Readability: ≥80% of outputs at ≤8th-grade reading level (Flesch-Kincaid). Faithfulness: ≥85% factual fidelity on the human-annotated test set. Hallucination rate: <10% of outputs contain a clinically meaningful unsupported claim. Multi-agent vs. single-agent baseline: ≥15% absolute improvement in faithfulness score. Verifier agent agreement with human annotators: ≥80%.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Exploration & Preprocessing | Develop ingestion pipelines for FHIR-formatted medical datasets and implement rigorous data cleaning and outlier detection protocols. |
| **October** | Feature Engineering & Baseline Modeling | Construct the multi-agent orchestration layer and establish baseline model performance using single-agent configurations for comparison. |
| **November** | Model Optimization & Evaluation | Execute iterative hyperparameter tuning, perform rigorous faithfulness testing, and optimize agent decision paths based on validation scores. |
| **December** | Insights, Deliverables & Presentation | Finalize the technical report, package the Hugging Face Space demonstration, and prepare actionable business recommendations for the stakeholders. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** MTSamples, Synthea, MedQuAD, and PLABA datasets.  
**Format:** FHIR (JSON-based clinical standards)  
**Size:** under 1gb  
**Location:** [Internal project repository and public research mirrors]  

### Key Details
- Publicly available de-identified clinical text including MTSamples transcribed medical reports, Synthea synthetic patient records, MedQuAD consumer health Q&A, and the PLABA plain-language adaptation corpus, in FHIR format.
- Data requires strict handling of medical terminology, with preprocessing focused on converting structured FHIR resources into readable text blocks for LLM consumption.

---

## 🛠️ Suggested Approach
**ML Problem Type:** NLP & RAG / Multi-Agent Systems  
**Recommended Libraries:**
- Open-source LLMs (Llama-3, Mistral, Gemma)
- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- Python
- GitHub
- Colab
- Hugging Face Inference API
- Hugging Face Spaces
**Evaluation Metrics:** Flesch-Kincaid Readability Score, Factual Fidelity (Faithfulness), Hallucination Rate, and Verifier-Human Agreement (%).

---

## 📚 Resources to Get Started
The following resources will help your team understand the problem space and potential technical approaches for this project:
**Background Reading:**
- Research on LLM hallucinations in clinical settings and the impact of health literacy on patient outcomes.
**Technical Tutorials:**
- Documentation for LangGraph or similar multi-agent orchestration frameworks.
**Code Examples:**
- Hugging Face Transformers documentation and reference implementations for RAG-based architectures.

---

## 🤝 How We'll Work Together
**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Email and designated team Slack/Teams channel  
**Response time:** 48 hours for non-urgent technical inquiries  
**Recommended Tools:**
- **Coding:** Google Colab Free Tier  
- **Collaboration:** GitHub, Notion  
- **Virtual Meetings:** Zoom, Google Meet  

---

## 🚀 Getting Started
1. **Review this overview document** and note any questions for our first meeting.
2. **Begin reviewing the dataset** using the link provided in the Dataset section.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects).

I'm excited to work with you!

---

## ❓ Questions?
Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
