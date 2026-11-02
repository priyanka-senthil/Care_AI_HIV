# CareAI: HIV Patient Information Assistant

CareAI answers questions from people living with HIV using only verified public health sources
(CDC, NIH HIVinfo, HIV.gov, and MedlinePlus). It combines an agentic workflow, a Retrieval-Augmented
Generation (RAG) pipeline, Google Gemini on Vertex AI, and a supervised fine-tuning step.

> CareAI gives general information only. It does not diagnose and does not replace an HIV care team.

## How it works

```
User message
   │
   ▼
1. Triage ──────── rule list + Gemini Flash ──► emergency / crisis ──► fixed safety reply (911 / 988)
   │
   ▼
2. Router ──────── Gemini Flash ──────────────► out of scope ──► polite redirect
   │  (intent + standalone search query)
   ▼
3. Retriever ───── embeddings + keyword score ─► no evidence ──► "ask your care team"
   │  (top-k chunks above a score threshold)
   ▼
4. Specialist generator ── tuned Gemini (or Gemini 2.5 Pro) with an intent-specific instruction
   │
   ▼
5. Safety check ── citation and rule check + Gemini grounding check ──► fails: regenerate once, then fall back
   │
   ▼
6. Urgent note (for example PEP timing) ──► answer with numbered citations
```

| Component | File | What it does |
|---|---|---|
| Triage | `careai/triage.py` | Two layers: fixed rules (cannot be skipped) and a model check. The more serious level wins. |
| Router | `careai/router.py` | Picks one of five intents and rewrites follow-up questions into standalone queries. |
| Specialist agents | `careai/agents.py` | Shared safety rules plus one instruction per intent (information, medication, emotional support, care navigation). |
| Orchestrator | `careai/orchestrator.py` | Runs the steps in order, handles early exits, retries, and records a trace of every step. |
| Ingestion | `careai/ingest.py`, `scripts/build_index.py` | Downloads trusted pages, removes menus and footers, splits text into overlapping chunks. |
| Vector index | `careai/vector_store.py`, `careai/retriever.py` | Stores embeddings; hybrid dense + keyword search with a score threshold and duplicate removal. |
| Grounding check | `careai/safety.py` | Second model call that lists any claim the evidence does not support. |
| Fine-tuning | `tuning/` | Builds the training set, runs the Vertex AI tuning job, and evaluates base vs. tuned. |
| App | `app.py` | Streamlit chat with sources, urgency banners, and an optional agent trace view. |

## Setup

### 1. Install
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Connect Google Cloud
1. Create or choose a Google Cloud project with billing turned on.
2. Turn on the Vertex AI API: `gcloud services enable aiplatform.googleapis.com`
3. Sign in: `gcloud auth application-default login`
4. Copy `.env.example` to `.env` and set `GOOGLE_CLOUD_PROJECT`.

### 3. Build the knowledge base
```bash
python scripts/build_index.py
```
This downloads the pages in `data/sources.yaml`, creates embeddings, and saves them in `data/index/`.
Pages that fail are listed in `data/index/build_report.json`. You can also put trusted `.txt`, `.md`,
or `.pdf` files in `data/local_docs/` before running the script.

### 4. Run the app
```bash
streamlit run app.py
```

### Offline demo (no Google Cloud)
Set `CAREAI_BACKEND=mock` to run with a simple offline model. This is useful for testing the workflow,
but answers will be basic.

## Fine-tuning Gemini on Vertex AI

RAG supplies the facts. Fine-tuning teaches the behavior: cite every fact, refuse when the evidence
does not cover the question, refer personal medicine decisions to the care team, and use plain language.

```bash
# 1. Create training data from the knowledge base (Gemini 2.5 Pro writes the examples)
python tuning/build_dataset.py --max-chunks 300

# 2. Review tuning/data/review.csv and remove weak examples

# 3. Upload to Cloud Storage
gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1
gcloud storage cp tuning/data/*.jsonl gs://YOUR_BUCKET/careai/

# 4. Start the tuning job (base model: gemini-2.5-flash, set in config.yaml)
python tuning/run_tuning.py --train gs://YOUR_BUCKET/careai/train.jsonl \
                            --validation gs://YOUR_BUCKET/careai/validation.jsonl

# 5. Put the printed endpoint in .env as CAREAI_TUNED_MODEL, then compare
python tuning/evaluate.py --label base       # run before setting CAREAI_TUNED_MODEL
python tuning/evaluate.py --label tuned      # run after
```
Results are saved in `tuning/results/`. Tuning has a cost on Google Cloud; check Vertex AI pricing first.

The dataset has three kinds of examples:
- **Grounded:** a realistic patient question with an ideal cited answer.
- **Safety:** a question that asks for a personal medicine decision, answered with general facts and a referral.
- **Unanswerable:** a question paired with unrelated evidence, answered with the standard "I could not find this" reply.

## Tests
```bash
pytest -q
```
The tests run offline. They cover HTML cleaning, chunking, the vector index, triage rules, output
safety rules, the full workflow (emergency, crisis, out of scope, grounded answer, urgent note,
no evidence), and the tuning data format.

## Safety design
- Emergency and crisis messages never reach the generator. A fixed reply gives 911 or 988.
- Rule-based triage runs even if the model is unavailable.
- Answers must cite evidence; answers that give personal medicine instructions are blocked.
- A grounding check compares every answer with the evidence; failures are regenerated once, then replaced with a safe referral.
- Only U.S. government health sources are used. Review `data/sources.yaml` before adding new ones.

## Project structure
```
careai-hiv/
├── app.py                  Streamlit chat app
├── config.yaml             Models, RAG settings, tuning settings
├── careai/                 Core package (triage, router, agents, retriever, safety, orchestrator)
├── scripts/build_index.py  Builds the knowledge base
├── data/sources.yaml       Trusted source list
├── tuning/                 Dataset builder, tuning job, evaluation set and script
└── tests/                  Offline tests
```
