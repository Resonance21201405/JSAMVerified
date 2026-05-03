# 🏗️ BIS Standards Recommendation Engine

**BIS Hackathon 2026 — Track: AI / Retrieval Augmented Generation (RAG)**

An AI-powered system that helps Indian MSEs instantly identify applicable Bureau of Indian Standards (BIS) standards from product descriptions.

---

## 📐 System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           QUERY EXPANSION AGENT             │
│  - Domain synonym injection (OPC, PPC, PSC) │
│  - IS-number extraction & exact-match boost │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐   ┌──────────────────┐
│  SPARSE      │   │  DENSE (optional)│
│  RETRIEVER   │   │  RETRIEVER       │
│  TF-IDF      │   │  sentence-       │
│  (1-3 ngram) │   │  transformers +  │
│              │   │  FAISS           │
└──────┬───────┘   └──────┬───────────┘
       │                  │
       └──────────┬───────┘
                  ▼
        ┌─────────────────┐
        │  HYBRID FUSION  │
        │  0.7×sparse +   │
        │  0.3×dense +    │
        │  exact-ID boost │
        │  title boost    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  LLM RERANKER   │
        │  (Claude Haiku) │
        │  Top-10 → Top-5 │
        │  + rationale    │
        └────────┬────────┘
                 │
                 ▼
        Retrieved Standards
        [IS XXX: YYYY, ...]
```

### Why this design achieves high accuracy + low latency:

| Component | Role | Latency |
|-----------|------|---------|
| TF-IDF (1-3 ngram) | High-precision keyword match; catches IS numbers, grades, material names | ~5ms |
| Query expansion | Bridges vocabulary gap (e.g. "OPC" → "ordinary portland cement") | <1ms |
| Exact-ID boost | Perfect recall when query mentions IS number directly | <1ms |
| Dense FAISS (optional) | Semantic similarity for paraphrased queries | ~20ms |
| Claude Haiku reranking | Intelligent reranking + rationale, anti-hallucination filter | ~800ms |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional: for dense retrieval (better accuracy)
pip install sentence-transformers faiss-cpu
```

### 2. Set up API key (for LLM reranking)

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 3. Build the index

```bash
# Place SP21 PDF at data/dataset.pdf, then:
python setup.py --pdf data/dataset.pdf
```

### 4. Run inference (judges use this)

```bash
python inference.py \
    --input data/public_test_set.json \
    --output data/results.json
```

### 5. Evaluate results

```bash
python eval_script.py --results data/results.json
```

### 6. Launch web UI (optional)

```bash
pip install streamlit
streamlit run app.py
```

---

## 📁 Project Structure

```
.
├── inference.py            # ← Judge entry point (mandatory)
├── setup.py                # ← One-time index builder
├── app.py                  # ← Streamlit web UI
├── eval_script.py          # ← Official evaluation script
├── requirements.txt
├── src/
│   ├── ingest.py           # PDF parsing + chunking + index building
│   ├── retriever.py        # Hybrid sparse+dense retriever
│   └── agent.py            # LLM reranker (Claude Haiku)
└── data/
    ├── dataset.pdf         # SP21 source document (place here)
    ├── public_test_set.json
    ├── results.json        # Output from inference.py
    └── index/              # Auto-built by setup.py
        ├── chunks.json
        ├── sparse_index.pkl
        ├── dense_index.pkl (optional)
        └── faiss.index (optional)
```

---

## 📊 Chunking Strategy

Each IS standard from SP 21 becomes one "chunk" containing:
- **Standard ID** (normalized, e.g. `IS 269: 1989`)
- **Full title** (multi-line title merged)
- **Scope sentence** (extracted from section 1)
- **Full standard text** (up to 200 lines)
- **Searchable field** = ID + title + scope + text (for TF-IDF)

This one-standard-per-chunk approach ensures:
- No cross-contamination between standards
- Perfect recall for exact-ID matches
- Efficient TF-IDF (529 chunks, 220K+ features)

---

## 🎯 Evaluation Results (Public Test Set)

| Metric | Score | Target |
|--------|-------|--------|
| Hit Rate @3 | **100%** | >80% |
| MRR @5 | **1.0000** | >0.7 |
| Avg Latency | **< 1s** (retrieval-only) | <5s |

---

## 🏛️ Hackathon Compliance

- ✅ `inference.py` at root with `--input` / `--output` args
- ✅ `eval_script.py` included (unmodified)
- ✅ Output JSON: `id`, `retrieved_standards`, `latency_seconds`
- ✅ Dataset integrity: SP21 is sole source of truth
- ✅ No hallucinated standards (LLM constrained to candidate list)
- ✅ Runs on standard hardware (no GPU required)
