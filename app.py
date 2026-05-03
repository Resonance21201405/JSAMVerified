"""
app.py — BIS Standards Recommendation Engine · Web Interface
=============================================================
Streamlit app with two modes:
  1. Single query  — type a product description, get top-5 IS standards
  2. JSON batch    — upload / paste a JSON file, run inference + eval

Usage:
    pip install streamlit
    streamlit run app.py

The app calls inference.py pipeline directly (no subprocess), so the
same retriever + LLM agent code runs here as in the CLI.
"""

import os
import sys
import json
import time
import tempfile
import logging
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Suppress noisy loggers in the UI
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIS Standards Engine",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #1a1a2e;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b7280;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
    border: 1px solid #dde3f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1a1a2e;
}

.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

.metric-pass { color: #059669 !important; }
.metric-fail { color: #dc2626 !important; }

.result-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid #4f46e5;
    transition: box-shadow 0.2s;
}

.result-card:hover { box-shadow: 0 4px 20px rgba(79,70,229,0.08); }

.result-rank {
    font-size: 0.7rem;
    font-weight: 600;
    color: #4f46e5;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.result-is {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #111827;
    margin: 0.15rem 0;
}

.result-title {
    font-size: 0.85rem;
    color: #374151;
    font-weight: 500;
}

.result-rationale {
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 0.4rem;
    font-style: italic;
    line-height: 1.5;
}

.hit-badge {
    display: inline-block;
    background: #d1fae5;
    color: #065f46;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.miss-badge {
    display: inline-block;
    background: #fee2e2;
    color: #991b1b;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #1a1a2e;
    border-bottom: 2px solid #4f46e5;
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}

.stTextArea textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

div[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stTextInput label,
div[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.sidebar-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e2e8f0 !important;
    margin-bottom: 0.2rem;
}

.sidebar-tagline {
    font-size: 0.72rem;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.query-row {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}

.query-id-badge {
    font-size: 0.68rem;
    font-weight: 600;
    color: #4f46e5;
    background: #eef2ff;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
}

.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(79,70,229,0.35) !important;
}

.eval-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}

.eval-table th {
    background: #f3f4f6;
    color: #374151;
    font-weight: 600;
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.eval-table td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid #f3f4f6;
    color: #374151;
    vertical-align: top;
}

.eval-table tr:hover td { background: #f9fafb; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading BIS index…")
def load_agent(index_dir: str, llm_backend: str, groq_model: str, groq_key: str):
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    from src.retriever import BISRetriever
    from src.agent import BISAgent

    retriever = BISRetriever(index_dir=index_dir)
    agent = BISAgent(
        retriever,
        llm_backend=llm_backend,
        groq_model=groq_model,
    )
    return agent


# ── Eval helper ───────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    import re
    s = re.sub(r":\s*\d{4}", "", s)
    return re.sub(r"[^A-Z0-9]", "", s.upper())

def _match(expected: str, retrieved: str) -> bool:
    if _norm(expected) == _norm(retrieved):
        return True
    e_base = re.sub(r"[^A-Z0-9]", "", re.sub(r"\(.*?\)", "", expected).upper())
    r_base = re.sub(r"[^A-Z0-9]", "", re.sub(r"\(.*?\)", "", retrieved).upper())
    return e_base == r_base and "PART" not in _norm(expected)

import re

def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    hit3 = mrr5 = lat = 0.0
    for r in results:
        exp = r.get("expected_standards", [])
        ret = r.get("retrieved_standards", [])
        lat += r.get("latency_seconds", 0)
        if exp and any(_match(e, rv) for e in exp for rv in ret[:3]):
            hit3 += 1
        for rank, rv in enumerate(ret[:5], 1):
            if exp and any(_match(e, rv) for e in exp):
                mrr5 += 1 / rank
                break
    return {
        "hit3": round(hit3 / total * 100, 2) if total else 0,
        "mrr5": round(mrr5 / total, 4) if total else 0,
        "avg_lat": round(lat / total, 3) if total else 0,
        "total": total,
        "hits": int(hit3),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏛️ BIS Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Standards Recommendation System</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Configuration**")

    index_dir = st.text_input("Index directory", value="data/index",
                               help="Path to the folder containing chunks.json and sparse_merged.pkl")

    llm_backend = st.selectbox("LLM backend", ["groq", "none", "anthropic"],
                                help="groq = free Llama 3 via Groq API\nnone = pure TF-IDF (no LLM)")

    groq_model = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        help="70b = best accuracy (~800ms)\n8b-instant = fastest (~200ms)",
        disabled=(llm_backend != "groq"),
    )

    groq_key = st.text_input(
        "Groq API key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at console.groq.com",
        disabled=(llm_backend != "groq"),
    )

    top_k = st.slider("Results per query", min_value=3, max_value=10, value=5)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#475569; line-height:1.8;">
    <b style="color:#94a3b8;">Quick start</b><br>
    1. Set index directory<br>
    2. Add Groq API key<br>
    3. Choose Single Query or Batch mode<br>
    4. Click Run
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 11])
with col_title:
    st.markdown('<div class="main-title">BIS Standards Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Bureau of Indian Standards · SP 21 · Building Materials</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Mode tabs ─────────────────────────────────────────────────────────────────

tab_single, tab_batch = st.tabs(["  🔍  Single Query  ", "  📂  Batch / JSON Evaluation  "])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE QUERY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_single:
    st.markdown('<div class="section-header">Describe your product or compliance need</div>', unsafe_allow_html=True)

    example_queries = [
        "Select an example…",
        "We manufacture 33 Grade Ordinary Portland Cement. Which BIS standard covers chemical and physical requirements?",
        "Looking for corrugated asbestos cement sheets used for roofing and cladding.",
        "Which standard covers hollow and solid lightweight concrete masonry blocks?",
        "We produce TMT ribbed bars for reinforcing concrete slabs. What IS standard governs yield strength?",
        "Specification for bitumen felt used as a waterproofing layer under roofing tiles.",
        "Which Indian Standard covers gun-metal gate valves used in plumbing?",
        "We supply uPVC pipes for potable water supply. Which BIS standard applies?",
        "Standard for glass wool slabs used as thermal insulation in walls and roofs.",
    ]

    selected_example = st.selectbox("Load an example query", example_queries)

    query_text = st.text_area(
        "Query",
        value="" if selected_example == example_queries[0] else selected_example,
        height=110,
        placeholder="Describe the product, material, or compliance requirement…",
        label_visibility="collapsed",
    )

    run_single = st.button("🔍  Find Standards", key="run_single", use_container_width=False)

    if run_single:
        if not query_text.strip():
            st.warning("Please enter a query.")
        else:
            try:
                with st.spinner("Loading index and running pipeline…"):
                    agent = load_agent(index_dir, llm_backend, groq_model, groq_key)
                    t0 = time.perf_counter()
                    results = agent.answer(query_text.strip(), top_k=top_k)
                    elapsed = round(time.perf_counter() - t0, 3)

                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:1rem; margin:1.2rem 0;">
                    <span style="font-size:0.8rem; color:#6b7280;">
                        ⚡ {elapsed:.2f}s &nbsp;·&nbsp; {len(results)} results
                    </span>
                </div>
                """, unsafe_allow_html=True)

                rank_colors = ["#4f46e5", "#7c3aed", "#9333ea", "#a855f7", "#c084fc"]
                for i, r in enumerate(results):
                    color = rank_colors[min(i, len(rank_colors)-1)]
                    rationale_html = f'<div class="result-rationale">"{r.get("rationale", "")}"</div>' if r.get("rationale") else ""
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:{color}">
                        <div class="result-rank">#{i+1}</div>
                        <div class="result-is">{r['std_id']}</div>
                        <div class="result-title">{r.get('title','').title()}</div>
                        {rationale_html}
                    </div>
                    """, unsafe_allow_html=True)

                # Download button
                output_json = json.dumps([{
                    "id": "Q1",
                    "query": query_text.strip(),
                    "retrieved_standards": [r["std_id"] for r in results],
                    "latency_seconds": elapsed,
                }], indent=2)
                st.download_button(
                    "⬇  Download result JSON",
                    data=output_json,
                    file_name="bis_result.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure the index directory exists and contains `chunks.json` and `sparse_merged.pkl`.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH / JSON EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.markdown('<div class="section-header">Upload or paste a query JSON file</div>', unsafe_allow_html=True)

    col_upload, col_paste = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("**Upload JSON file**")
        uploaded = st.file_uploader(
            "Drop JSON here",
            type=["json"],
            label_visibility="collapsed",
            help='JSON array of objects with "id" and "query" fields. Optional: "expected_standards".',
        )

    with col_paste:
        st.markdown("**Or paste JSON**")
        pasted = st.text_area(
            "JSON content",
            height=180,
            label_visibility="collapsed",
            placeholder='[{"id": "Q1", "query": "...", "expected_standards": ["IS 269: 1989"]}, ...]',
        )

    # Parse input
    queries_data = None
    parse_error  = None

    if uploaded:
        try:
            queries_data = json.loads(uploaded.read().decode("utf-8"))
        except Exception as e:
            parse_error = str(e)
    elif pasted.strip():
        try:
            queries_data = json.loads(pasted.strip())
        except Exception as e:
            parse_error = str(e)

    if parse_error:
        st.error(f"JSON parse error: {parse_error}")

    if queries_data:
        st.markdown(f"**{len(queries_data)} queries loaded** — preview:")
        with st.expander("Show loaded queries", expanded=False):
            for q in queries_data[:5]:
                exp_str = ", ".join(q.get("expected_standards", [])) or "—"
                st.markdown(f"""
                <div class="query-row">
                    <span class="query-id-badge">{q.get('id','?')}</span>
                    <span style="font-size:0.85rem; color:#374151;">{q.get('query','')[:120]}{'…' if len(q.get('query',''))>120 else ''}</span>
                    <div style="font-size:0.75rem; color:#9ca3af; margin-top:4px;">Expected: {exp_str}</div>
                </div>
                """, unsafe_allow_html=True)
            if len(queries_data) > 5:
                st.caption(f"… and {len(queries_data)-5} more")

        st.markdown("")
        run_batch = st.button("▶  Run Inference + Evaluate", key="run_batch", use_container_width=False)

        if run_batch:
            try:
                with st.spinner(f"Loading index…"):
                    agent = load_agent(index_dir, llm_backend, groq_model, groq_key)

                results   = []
                progress  = st.progress(0, text="Running queries…")
                status_ph = st.empty()
                total_q   = len(queries_data)

                for i, item in enumerate(queries_data):
                    qid   = item.get("id", str(i+1))
                    query = item.get("query", "")
                    status_ph.markdown(f"<span style='font-size:0.8rem;color:#6b7280;'>Running [{i+1}/{total_q}] {qid}: {query[:60]}…</span>", unsafe_allow_html=True)

                    t0 = time.perf_counter()
                    try:
                        retrieved = agent.answer(query, top_k=top_k)
                    except Exception as e:
                        retrieved = []
                        st.warning(f"[{qid}] failed: {e}")
                    elapsed = round(time.perf_counter() - t0, 3)

                    result = {
                        "id":                  qid,
                        "query":               query,
                        "retrieved_standards": [r["std_id"] for r in retrieved],
                        "latency_seconds":     elapsed,
                        "_retrieved_full":     retrieved,
                    }
                    if "expected_standards" in item:
                        result["expected_standards"] = item["expected_standards"]
                    results.append(result)
                    progress.progress((i+1)/total_q, text=f"Completed {i+1}/{total_q}")

                status_ph.empty()
                progress.empty()

                st.markdown("---")

                # ── Metrics ────────────────────────────────────────────────
                has_expected = any("expected_standards" in r for r in results)

                if has_expected:
                    metrics = compute_metrics(results)
                    st.markdown('<div class="section-header">Evaluation Results</div>', unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        pass_class = "metric-pass" if metrics["hit3"] > 80 else "metric-fail"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value {pass_class}">{metrics['hit3']:.1f}%</div>
                            <div class="metric-label">Hit Rate @3</div>
                            <div style="font-size:0.68rem;color:#9ca3af;margin-top:4px;">target &gt; 80%  {'✓' if metrics['hit3']>80 else '✗'}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        pass_class = "metric-pass" if metrics["mrr5"] > 0.7 else "metric-fail"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value {pass_class}">{metrics['mrr5']:.4f}</div>
                            <div class="metric-label">MRR @5</div>
                            <div style="font-size:0.68rem;color:#9ca3af;margin-top:4px;">target &gt; 0.70  {'✓' if metrics['mrr5']>0.7 else '✗'}</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        pass_class = "metric-pass" if metrics["avg_lat"] < 5 else "metric-fail"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value {pass_class}">{metrics['avg_lat']:.2f}s</div>
                            <div class="metric-label">Avg Latency</div>
                            <div style="font-size:0.68rem;color:#9ca3af;margin-top:4px;">target &lt; 5s  {'✓' if metrics['avg_lat']<5 else '✗'}</div>
                        </div>""", unsafe_allow_html=True)
                    with c4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color:#1a1a2e;">{metrics['hits']}/{metrics['total']}</div>
                            <div class="metric-label">Queries Hit</div>
                            <div style="font-size:0.68rem;color:#9ca3af;margin-top:4px;">out of {metrics['total']} total</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("")

                # ── Per-query results table ────────────────────────────────
                st.markdown('<div class="section-header">Per-query Results</div>', unsafe_allow_html=True)

                rows_html = ""
                for r in results:
                    exp  = r.get("expected_standards", [])
                    ret  = r.get("retrieved_standards", [])
                    lat  = r.get("latency_seconds", 0)

                    if exp:
                        hit = any(_match(e, rv) for e in exp for rv in ret[:3])
                        badge = '<span class="hit-badge">HIT@3</span>' if hit else '<span class="miss-badge">MISS</span>'
                    else:
                        badge = ""

                    ret_html = "".join(f'<div style="font-size:0.8rem;color:#374151;margin:1px 0;">{s}</div>' for s in ret[:3])
                    exp_html = "".join(f'<div style="font-size:0.8rem;color:#065f46;margin:1px 0;">{s}</div>' for s in exp) if exp else '<div style="font-size:0.8rem;color:#9ca3af;">—</div>'

                    rows_html += f"""
                    <tr>
                        <td><span style="font-size:0.75rem;font-weight:600;color:#4f46e5;">{r['id']}</span></td>
                        <td style="max-width:280px;word-break:break-word;">{r['query'][:100]}{'…' if len(r['query'])>100 else ''}</td>
                        <td>{ret_html}</td>
                        <td>{exp_html}</td>
                        <td>{badge}</td>
                        <td style="text-align:right;font-variant-numeric:tabular-nums;color:#6b7280;">{lat:.2f}s</td>
                    </tr>"""

                st.markdown(f"""
                <table class="eval-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Query</th>
                            <th>Retrieved (top 3)</th>
                            <th>Expected</th>
                            <th>Status</th>
                            <th style="text-align:right">Latency</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

                # ── Downloads ─────────────────────────────────────────────
                st.markdown("")
                col_dl1, col_dl2 = st.columns(2)

                # Clean output JSON (remove internal _retrieved_full key)
                clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
                output_json = json.dumps(clean_results, indent=2, ensure_ascii=False)

                with col_dl1:
                    st.download_button(
                        "⬇  Download output.json",
                        data=output_json,
                        file_name="output.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                if has_expected:
                    eval_report = {
                        "hit_rate_at_3":    metrics["hit3"],
                        "mrr_at_5":         metrics["mrr5"],
                        "avg_latency_secs": metrics["avg_lat"],
                        "total_queries":    metrics["total"],
                        "hits":             metrics["hits"],
                        "pass_hit3":        metrics["hit3"] > 80,
                        "pass_mrr5":        metrics["mrr5"] > 0.7,
                        "pass_latency":     metrics["avg_lat"] < 5,
                    }
                    with col_dl2:
                        st.download_button(
                            "⬇  Download eval_report.json",
                            data=json.dumps(eval_report, indent=2),
                            file_name="eval_report.json",
                            mime="application/json",
                            use_container_width=True,
                        )

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.75rem; color:#9ca3af; padding:0.5rem 0;">
    BIS Standards Recommendation Engine &nbsp;·&nbsp;
    SP 21 Building Materials &nbsp;·&nbsp;
    TF-IDF + Groq LLM Pipeline
</div>
""", unsafe_allow_html=True)