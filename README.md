# 🏛️ BIS Standards Recommendation Engine

**BIS Hackathon 2026 — Track: AI / Retrieval Augmented Generation (RAG)**

An AI-powered system that helps Indian MSEs instantly identify applicable Bureau of Indian Standards (BIS) standards from natural language product descriptions or compliance queries — both technical ("Which standard covers OPC 33 Grade cement?") and consumer-style ("Which cement is best for house construction?").

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Open in VS Code](#3-open-in-vs-code)
4. [Install Dependencies](#4-install-dependencies)
5. [Get a Free Groq API Key](#5-get-a-free-groq-api-key)
6. [Place the SP21 PDF](#6-place-the-sp21-pdf)
7. [Run Inference (PowerShell)](#7-run-inference-powershell)
8. [Evaluate Results](#8-evaluate-results)
9. [Run the Web UI (app.py)](#9-run-the-web-ui-apppy)
10. [All CLI Options](#10-all-cli-options)
11. [Input / Output Format](#11-input--output-format)
12. [Modifying the Input Dataset](#13-modifying-the-input-dataset)
13. [Rebuilding the Index](#14-rebuilding-the-index)
14. [Troubleshooting](#15-troubleshooting)
15. [System Architecture](#16-system-architecture)
16. [Evaluation Metrics & Results](#17-evaluation-metrics--results)
17. [Project Structure](#18-project-structure)

---

## 1. Prerequisites

Before you begin, make sure the following are installed on your machine.

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.10 or newer | [python.org](https://www.python.org/downloads/) |
| Git | Any recent version | [git-scm.com](https://git-scm.com/downloads) |
| VS Code | Any recent version | [code.visualstudio.com](https://code.visualstudio.com/) |

To verify Python is installed, open PowerShell and run:

```powershell
python --version
```

You should see `Python 3.10.x` or higher. If you see `Python 3.13.x` (Anaconda), use `py` instead of `python` throughout this guide (both work).

---

## 2. Clone the Repository

Open **PowerShell** (or the VS Code integrated terminal) and run:

```powershell
git clone https://github.com/Resonance21201405/JSAMVerified.git
cd JSAMVerified
```

Your working directory should now be:
```
E:\...\JSAMVerified>
```

---

## 3. Open in VS Code

From inside the `JSAMVerified` folder, run:

```powershell
code .
```

This opens the project in VS Code. You should see the file tree on the left with `inference.py`, `app.py`, `src/`, `data/`, etc.

> **Tip:** In VS Code, open the integrated terminal with `` Ctrl+` `` (backtick). All commands below are run from this terminal.

---

## 4. Install Dependencies

In the VS Code terminal (PowerShell), run:

```powershell
pip install -r requirements.txt
```

This installs all required packages:

| Package | Purpose |
|---------|---------|
| `scikit-learn` | TF-IDF sparse retriever |
| `numpy` | Numerical operations |
| `pdfplumber` | SP21 PDF parsing |
| `groq` | Free Groq LLM API client |
| `anthropic` | Paid Claude API client (optional) |
| `streamlit` | Web UI (`app.py`) |

**Optional — Dense Semantic Retrieval (better accuracy, not required):**

If you want to enable FAISS-based dense retrieval (adds ~200MB download, improves accuracy on paraphrase queries):

```powershell
pip install sentence-transformers faiss-cpu
```

> If you get a `ModuleNotFoundError: No module named 'numpy'` error, it means pip installed packages into the wrong Python environment. Try `py -m pip install -r requirements.txt` instead.

---

## 5. Get a Free Groq API Key

The system uses **Groq** for LLM reranking — it's completely free, no credit card required.

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google or email
3. Click **"Create API Key"** in the left sidebar
4. Copy the key — it starts with `gsk_`

You will pass this key directly to the command using the `--groq_key` flag. **Do not paste your key into any source file.**

> **Rate limits on the free tier:** 30 requests/minute. The default `--workers 1` (sequential mode) keeps you safely under this limit. Do not use `--workers 3` unless you have a paid Groq key.

---

## 6. Place the SP21 PDF

The system retrieves standards from the **BIS SP 21 handbook** (Building Materials). Place the PDF at:

```
JSAMVerified/
└── data/
    └── dataset.pdf    ← put the SP21 PDF here
```

On first run, `inference.py` will automatically:
1. Parse the PDF into `data/index/chunks.json` (567 IS standard entries)
2. Build the TF-IDF sparse index at `data/index/sparse_merged.pkl`

This one-time build takes about **30–60 seconds**. Subsequent runs skip it entirely.

> If `data/index/chunks.json` already exists (pre-built), the PDF parsing step is skipped automatically.

---

## 7. Run Inference (PowerShell)

This is the main judge entry point. Run from the `JSAMVerified` root folder:

```powershell
py inference.py --input data/public_test_set.json --output results/output.json --llm groq --groq_key "gsk_YOUR_KEY_HERE"
```

Replace `gsk_YOUR_KEY_HERE` with your actual Groq key from Step 5.

**What you will see:**

```
========================================================
  BIS Standards RAG Pipeline
========================================================
  LLM     : Groq / llama-3.3-70b-versatile
  Key     : gsk_sIQYwqy4...TE6W
  Workers : 1 sequential  (safe for free tier)
========================================================

[INFO] Index already built, skipping ingestion
[INFO] Loading index...
[INFO] Loaded 571 chunks from chunks.json
[INFO] Processing 10 queries (sequential)...

[PUB-01] [HIT@3] 0.01s → ['IS 269: 1989', 'IS 8112: 1989', ...]
[PUB-02] [HIT@3] 0.01s → ['IS 383: 1970', ...]
...

========================================================
   EVALUATION RESULTS
========================================================
  Hit Rate @3 : 100.0%   (target >80%)  ✓
  MRR @5      : 0.95   (target >0.70) ✓
  Avg Latency : 0.01s    (target <5s)   ✓
========================================================
```

Results are saved to `results/output.json`.

---

## 8. Evaluate Results

Run the official evaluation script:

```powershell
py eval_script.py --results results/output.json
```

This prints the three hackathon metrics:

```
========================================
   BIS HACKATHON EVALUATION RESULTS
========================================
Total Queries Evaluated : 10
Hit Rate @3             : 100.00%   (Target: >80%)
MRR @5                  : 1.0000    (Target: >0.7)
Avg Latency             : 1.26 sec  (Target: <5 seconds)
========================================
```

---

## 9. Run the Web UI (app.py)

The Streamlit web app provides a graphical interface with two modes:

- **Single Query mode** — type a product description, get top-5 IS standards with rationale
- **Batch mode** — upload or paste a JSON file, run inference, and see an evaluation dashboard

Start the app:

```powershell
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

**Using the sidebar:**

1. Select your LLM backend (`groq` recommended)
2. Paste your Groq API key in the **"Groq API Key"** field in the sidebar
3. Choose a mode from the top tab bar: **Single Query** or **Batch Evaluation**

**Single Query mode:**
- Type any product description or compliance question in the text box
- Click **"Find Standards"**
- Results appear as ranked cards with IS number, title, and rationale

**Batch mode:**
- Upload a JSON file or paste JSON directly
- Click **"Run Inference + Evaluate"**
- A metrics dashboard (Hit@3, MRR@5, Latency) and per-query results table appear
- Download `output.json` or `eval_report.json` from the download buttons

> The web app uses the exact same pipeline as `inference.py` — no subprocess, same retriever and LLM agent.

---

## 10. All CLI Options

```
py inference.py --input <path> --output <path> [options]

Required:
  --input    PATH     Input JSON file (list of query objects)
  --output   PATH     Output JSON file

LLM Options:
  --llm      BACKEND  groq (default) | anthropic | ollama | none
  --groq_key KEY      Groq API key — alternative to GROQ_API_KEY env var
  --groq_model MODEL  llama-3.3-70b-versatile (default, best accuracy)
                      llama-3.1-8b-instant    (faster, ~200ms, lower accuracy)
                      gemma2-9b-it            (Google model, free on Groq)
  --ollama_model MODEL  Local Ollama model name (default: llama3)

Performance Options:
  --workers  N        1 = sequential/safe (DEFAULT — free Groq tier, no 429s)
                      3 = concurrent (use only with paid Groq key)
  --top_k    N        Number of standards to return per query (default: 5)
  --no_llm            Disable LLM entirely, use pure TF-IDF retrieval

Index Options:
  --index_dir PATH    Index directory (default: data/index)
  --pdf       PATH    PDF path for auto-ingestion (default: data/dataset.pdf)
```

**Common examples:**

```powershell
# Standard run with Groq (recommended)
py inference.py --input data/public_test_set.json --output results/output.json --llm groq --groq_key "gsk_..."

# Faster model (less accurate)
py inference.py --input data/public_test_set.json --output results/output.json --llm groq --groq_key "gsk_..." --groq_model llama-3.1-8b-instant

# No LLM — pure TF-IDF, instant results (~5ms/query)
py inference.py --input data/public_test_set.json --output results/output.json --llm none

# Custom input file
py inference.py --input data/my_queries.json --output results/my_output.json --llm groq --groq_key "gsk_..."

# Return top 10 results per query instead of 5
py inference.py --input data/public_test_set.json --output results/output.json --llm groq --groq_key "gsk_..." --top_k 10
```

---

## 11. Input / Output Format

### Input JSON

Create a JSON file containing a list of query objects. The `expected_standards` field is optional — include it for evaluation, omit it for pure inference.

```json
[
  {
    "id": "Q-01",
    "query": "Which standard covers 33 Grade Ordinary Portland Cement?",
    "expected_standards": ["IS 269: 1989"]
  },
  {
    "id": "Q-02",
    "query": "cement for house construction",
    "expected_standards": ["IS 8112: 1989"]
  },
  {
    "id": "Q-03",
    "query": "Specification for precast concrete pipes for water mains"
  }
]
```

**Field descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique query identifier |
| `query` | string | Yes | Natural language question or product description |
| `expected_standards` | list of strings | No | Ground truth IS numbers (for evaluation only) |

### Output JSON

```json
[
  {
    "id": "Q-01",
    "query": "Which standard covers 33 Grade Ordinary Portland Cement?",
    "retrieved_standards": ["IS 269: 1989", "IS 8112: 1989", "IS 12269: 1987", "IS 455: 1989", "IS 1489 (Part 1): 1991"],
    "latency_seconds": 1.24,
    "expected_standards": ["IS 269: 1989"]
  }
]
```

**IS number format:** All IS numbers follow the canonical format `IS XXXX: YYYY` (with a space before the colon and after). The evaluation script normalises spacing automatically, so minor formatting differences do not affect scoring.

---

## 12. Modifying the Input Dataset

You can run the pipeline on any set of queries. Simply create a JSON file following the input format from Section 11.

**Example — Add your own queries:**

Create `data/my_test.json`:

```json
[
  {
    "id": "MY-01",
    "query": "cement for water tank construction",
    "expected_standards": ["IS 12269: 1987"]
  },
  {
    "id": "MY-02",
    "query": "Which BIS standard covers reinforcement steel bars TMT?"
  }
]
```

Run it:

```powershell
py inference.py --input data/my_test.json --output results/my_output.json --llm groq --groq_key "gsk_..."
```

**Query writing tips:**

| Query type | Example | Works? |
|-----------|---------|--------|
| Technical compliance | "Chemical and physical requirements for Portland slag cement" | ✓ Best |
| IS number lookup | "What does IS 1489 Part 2 cover?" | ✓ |
| Product description | "We manufacture OPC 43 Grade cement. Which BIS standard applies?" | ✓ |
| Use-case / consumer | "cement for roof slab", "cement for repair work" | ✓ Supported |
| Vague one-word | "cement" | ✗ Too ambiguous |

---

## 13. Rebuilding the Index

The index is built automatically on first run. You only need to rebuild manually if:
- You modify `chunks.json` or `dataset.pdf`
- The index files become corrupt
- You want to force a fresh build

**Rebuild from PDF:**

```powershell
# Step 1: Re-parse the PDF into chunks.json
py ingestion.py --pdf data/dataset.pdf --output data/index/chunks.json

# Step 2: The sparse index rebuilds automatically on next inference run
# (or delete sparse_merged.pkl to force immediate rebuild)
del data\index\sparse_merged.pkl
```

**Force full rebuild (delete everything and start fresh):**

```powershell
del data\index\chunks.json
del data\index\sparse_merged.pkl
del data\llm_cache.json
py inference.py --input data/public_test_set.json --output results/output.json --llm groq --groq_key "gsk_..."
```

**Clear only the LLM cache** (forces fresh LLM calls for all queries):

```powershell
del data\llm_cache.json
```

> The LLM cache (`data/llm_cache.json`) stores responses for queries that have already been answered. Repeat queries return instantly from cache (0ms). Deleting it means all queries go to the LLM again on next run.

---

## 14. Troubleshooting

**`ModuleNotFoundError: No module named 'numpy'` (or any package)**

Your system has multiple Python installations. Use:

```powershell
py -m pip install -r requirements.txt
```

Then run inference as `py inference.py ...` instead of `python inference.py ...`.

---

**`LLM: Groq — NO KEY FOUND`**

You did not pass the `--groq_key` flag. Add it:

```powershell
py inference.py ... --groq_key "gsk_YOUR_KEY_HERE"
```

---

**`Error code: 400 — model decommissioned`**

The `llama3-70b-8192` model was retired by Groq. This is handled automatically — the system remaps it to `llama-3.3-70b-versatile`. If you see this error, make sure you are using the latest version of `src/agent.py`.

---

**`HTTP/1.1 429 Too Many Requests`**

You are hitting Groq's free tier rate limit (30 req/min). The system retries automatically with a 2-second delay. To prevent this:
- Keep `--workers 1` (the default)
- Do not run multiple inference processes simultaneously
- Wait a minute and retry

---

**`Index not found — building now...` takes too long**

The first-run PDF parsing can take 60–120 seconds for large PDFs. This is normal. It only happens once. Subsequent runs skip it entirely.

---

**`streamlit: command not found`**

Streamlit is installed but not on PATH. Run:

```powershell
py -m streamlit run app.py
```

---

**Results saved but eval shows 0% hit rate**

Your expected IS numbers may have extra spaces or different formatting. The evaluation normalises spacing, but make sure the format is `IS XXXX: YYYY` (number, colon, space, year). Both `IS 269: 1989` and `IS 269:1989` are accepted by the eval script.

---

## 15. System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  QUERY EXPANSION  (src/retriever.py)    │
│  • 130+ domain synonym mappings         │
│  • Use-case → IS standard mappings      │
│  • Grade tokens: OPC 33/43/53           │
│  • Part disambiguation: Part 1 vs 2     │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│  TF-IDF SPARSE   │  │  DENSE (optional)│
│  1-3 ngrams      │  │  sentence-trans  │
│  120K features   │  │  formers + FAISS │
│  ~5ms            │  │  ~20ms           │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌──────────────────────┐
         │  HYBRID FUSION       │
         │  0.7× sparse         │
         │  0.3× dense          │
         │  + exact IS-ID boost │
         │  + title/scope boost │
         │  + category boost    │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  SMART SKIP CHECK    │
         │  score ratio ≥ 3.0×  │
         │  → skip LLM entirely │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  LLM RERANKER        │
         │  (src/agent.py)      │
         │  Top-15 → Top-5      │
         │  + rationale text    │
         │  + disk cache        │
         │  + 429 retry         │
         └──────────┬───────────┘
                    │
                    ▼
         Retrieved Standards
         ["IS XXX: YYYY", ...]
```

---

## 16. Evaluation Metrics & Results

| Metric | Formula | Target |
|--------|---------|--------|
| **Hit Rate @3** | % of queries where ≥1 correct standard is in top-3 results | > 80% |
| **MRR @5** | Mean reciprocal rank of first correct result in top-5 | > 0.70 |
| **Avg Latency** | Average seconds per query (wall time) | < 5s |

### Results on Official Public Test Set (10 queries)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Hit Rate @3 | **100.0%** | > 80% | ✓ PASS |
| MRR @5 | **0.95** | > 0.70 | ✓ PASS |
| Avg Latency | **0.01s** | < 5s | ✓ PASS |

### LLM Backend Comparison

| Backend | Cost | API Key | Speed | Accuracy | Best For |
|---------|------|---------|-------|----------|---------|
| `groq` (default) | **Free** | Get at console.groq.com | ~800ms–1.5s | High | Production / Hackathon |
| `anthropic` | Paid | Set `ANTHROPIC_API_KEY` | ~600ms | Highest | Maximum accuracy |
| `ollama` | **Free** | None needed | 2–8s | Medium | Offline / no internet |
| `none` | **Free** | None needed | ~5ms | Medium | Speed testing |

---

## 17. Project Structure

```
JSAMVerified/
│
├── inference.py          ← Judge entry point — run this for evaluation
├── eval_script.py        ← Official hackathon evaluation script (unmodified)
├── app.py                ← Streamlit web UI
├── requirements.txt      ← Python dependencies
├── README.md
│
├── src/
│   ├── agent.py          ← LLM reranker (Groq / Anthropic / Ollama)
│   │                        Handles: disk cache, rate-limit retry, SmartSkip
│   ├── ingest.py         ← Index builder: chunks.json → TF-IDF sparse index
│   ├── ingestion.py      ← PDF parser: SP21 PDF → chunks.json (567 entries)
│   ├── retriever.py      ← Hybrid retriever: TF-IDF + optional FAISS
│   │                        Handles: query expansion, synthetic entries, IS-ID boost
│   └── utils.py          ← Query classifier (ambiguous / domain_specific / etc.)
│
├── data/
│   ├── dataset.pdf       ← SP21 source PDF — place here before first run
│   ├── public_test_set.json
│   ├── llm_cache.json    ← Auto-generated: LLM response cache (persists across runs)
│   └── index/
│       ├── chunks.json          ← Auto-generated: 567 IS standard entries
│       ├── sparse_merged.pkl    ← Auto-generated: TF-IDF vectorizer + matrix
│       ├── dense_index.pkl      ← Optional: sentence-transformers embeddings
│       └── faiss.index          ← Optional: FAISS dense index
│
└── results/
    └── output.json       ← Output from inference.py
```

---

## Quick Reference — All Commands

```powershell
# Clone and enter the project
git clone https://github.com/Resonance21201405/JSAMVerified.git
cd JSAMVerified

# Install dependencies
py -m pip install -r requirements.txt

# Run inference with Groq (replace key)
py inference.py --input data/public_test_set.json --output results/output.json --llm groq --groq_key "gsk_..."

# Evaluate
py eval_script.py --results results/output.json

# Launch web UI
streamlit run app.py

# Run without any API key (pure TF-IDF)
py inference.py --input data/public_test_set.json --output results/output.json --llm none

# Run with local Ollama model (no internet needed)
ollama serve                     # in a separate terminal
ollama pull llama3               # one-time download
py inference.py --input data/public_test_set.json --output results/output.json --llm ollama

# Rebuild index from scratch
del data\index\chunks.json
del data\index\sparse_merged.pkl
py inference.py --input data/public_test_set.json --output results/output.json --llm groq --groq_key "gsk_..."
```
