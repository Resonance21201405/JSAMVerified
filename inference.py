"""
inference.py — BIS Standards Recommendation Engine  [RATE-LIMIT FIXED]
=======================================================================
Key fixes:
  - Default workers=1 (sequential) for Groq free tier — prevents 429 cascades
  - Default model=llama-3.3-70b-versatile — better accuracy than 8b-instant
  - Use workers=3 only if you have a paid Groq key with higher rate limits
  - API key loaded from env var (never hardcode in source)

Usage:
  # Free Groq key (sequential, safe)
  py inference.py --input data/public_test_set.json --output results/output.json --llm groq

  # Paid Groq key (concurrent, faster)
  py inference.py --input data/public_test_set.json --output results/output.json --llm groq --workers 3

  # No LLM (pure TF-IDF, instant)
  py inference.py --input data/public_test_set.json --output results/output.json --llm none
"""

import os
import sys
import json
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def setup_index_if_needed(index_dir: str = "data/index", pdf_path: str = "data/dataset.pdf"):
    chunks_path = os.path.join(index_dir, "chunks.json")
    sparse_path = os.path.join(index_dir, "sparse_merged.pkl")
    if os.path.exists(chunks_path) and os.path.exists(sparse_path):
        log.info("Index already built, skipping ingestion")
        return
    log.info("Index not found — building now...")
    from src.ingest import build_index
    build_index(pdf_path, index_dir)


def _process_query(args_tuple) -> tuple:
    idx, item, agent, top_k = args_tuple
    query_id = item.get("id", str(idx))
    query    = item.get("query", "")

    start = time.perf_counter()
    try:
        retrieved = agent.answer(query, top_k=top_k)
    except Exception as e:
        log.error(f"Error on {query_id}: {e}")
        retrieved = []
    elapsed = round(time.perf_counter() - start, 3)

    result = {
        "id":                  query_id,
        "query":               query,
        "retrieved_standards": [r["std_id"] for r in retrieved],
        "latency_seconds":     elapsed,
    }
    if "expected_standards" in item:
        result["expected_standards"] = item["expected_standards"]
    return idx, result


def main():
    parser = argparse.ArgumentParser(description="BIS Standards RAG Pipeline")
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--index_dir", default="data/index")
    parser.add_argument("--pdf",       default="data/dataset.pdf")
    parser.add_argument("--top_k",     type=int, default=5)
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Concurrent LLM threads.\n"
            "  1 = sequential (DEFAULT, safe for free Groq tier, no 429s)\n"
            "  3 = concurrent (use only with paid Groq key)"
        )
    )
    parser.add_argument("--llm", default="groq",
                        choices=["groq", "ollama", "anthropic", "none"])
    parser.add_argument(
        "--groq_model", default="llama-3.3-70b-versatile",
        help=(
            "Groq model:\n"
            "  llama-3.3-70b-versatile  (DEFAULT — best accuracy, ~800ms)\n"
            "  llama-3.1-8b-instant     (faster ~200ms, lower accuracy)\n"
        )
    )
    parser.add_argument("--ollama_model", default="llama3")
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument(
        "--groq_key", default="",
        help="Groq API key — alternative to setting GROQ_API_KEY env var. Get free at console.groq.com",
    )
    args = parser.parse_args()

    # Inject --groq_key into environment so all downstream code picks it up
    if args.groq_key:
        os.environ["GROQ_API_KEY"] = args.groq_key
        log.info(f"[Key] GROQ_API_KEY set from --groq_key ({args.groq_key[:12]}...)")

    if args.no_llm:
        args.llm = "none"

    # ── Banner ────────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  BIS Standards RAG Pipeline")
    print("=" * 56)
    if args.llm == "groq":
        key = os.environ.get("GROQ_API_KEY", "")
        if key:
            print(f"  LLM     : Groq / {args.groq_model}")
            print(f"  Key     : {key[:12]}...{key[-4:]}")
            if args.workers > 1:
                print(f"  Workers : {args.workers} concurrent  ⚠ ensure paid key")
            else:
                print(f"  Workers : 1 sequential  (safe for free tier)")
        else:
            print("  LLM: Groq — NO KEY FOUND")
            print("  Set env var:  $env:GROQ_API_KEY = 'gsk_...'")
            print("  Free key at: https://console.groq.com")
    elif args.llm == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        print(f"  LLM: Anthropic Claude  key: {(key[:12]+'...') if key else 'NOT SET'}")
    elif args.llm == "ollama":
        print(f"  LLM: Ollama / {args.ollama_model}")
    else:
        print("  LLM: DISABLED — pure TF-IDF")
    print("=" * 56 + "\n")

    # ── Setup ─────────────────────────────────────────────────────────────
    setup_index_if_needed(args.index_dir, args.pdf)

    from src.retriever import BISRetriever
    from src.agent import BISAgent

    log.info("Loading index...")
    retriever = BISRetriever(index_dir=args.index_dir)
    agent = BISAgent(
        retriever,
        llm_backend  = args.llm,
        ollama_model = args.ollama_model,
        groq_model   = args.groq_model,
    )

    with open(args.input, "r", encoding="utf-8") as f:
        queries = json.load(f)

    mode = f"{args.workers} concurrent worker(s)" if args.workers > 1 else "sequential"
    log.info(f"Processing {len(queries)} queries ({mode})...\n")

    # ── Run ───────────────────────────────────────────────────────────────
    total_start = time.perf_counter()
    results_map = {}
    work_items  = [(i, item, agent, args.top_k) for i, item in enumerate(queries)]

    if args.workers == 1 or args.llm == "none":
        # Sequential — safest, recommended for free Groq tier
        for work in work_items:
            idx, result = _process_query(work)
            results_map[idx] = result
            _log_result(queries[idx], result)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_query, w): w[0] for w in work_items}
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results_map[idx] = result
                    _log_result(queries[idx], result)
                except Exception as e:
                    log.error(f"Worker failed: {e}")

    total_elapsed = time.perf_counter() - total_start
    results = [results_map[i] for i in range(len(queries)) if i in results_map]

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg = total_elapsed / max(len(queries), 1)
    log.info(f"Saved {len(results)} results → {args.output}")
    log.info(f"Total: {total_elapsed:.2f}s  |  Avg/query: {avg:.2f}s")

    if results and "expected_standards" in queries[0]:
        _quick_eval(queries, results)


def _log_result(query_item: dict, result: dict):
    qid       = result["id"]
    retrieved = result["retrieved_standards"]
    elapsed   = result["latency_seconds"]
    if "expected_standards" in query_item:
        def norm(s): return str(s).replace(" ", "").lower()
        expected = set(norm(s) for s in query_item["expected_standards"])
        top3     = [norm(s) for s in retrieved[:3]]
        hit      = any(s in expected for s in top3)
        log.info(f"[{qid}] [{'HIT@3' if hit else 'MISS '}] {elapsed:.2f}s → {retrieved[:3]}")
        if not hit:
            log.info(f"         Expected: {query_item['expected_standards']}")
    else:
        log.info(f"[{qid}] {elapsed:.2f}s → {retrieved[:3]}")


def _quick_eval(queries: list, results: list):
    def norm(s): return str(s).replace(" ", "").lower()
    hits3, mrr5, total_lat = 0, 0.0, 0.0
    for q, r in zip(queries, results):
        expected  = set(norm(s) for s in q.get("expected_standards", []))
        retrieved = [norm(s) for s in r.get("retrieved_standards", [])]
        total_lat += r.get("latency_seconds", 0)
        if any(s in expected for s in retrieved[:3]):
            hits3 += 1
        for rank, s in enumerate(retrieved[:5], 1):
            if s in expected:
                mrr5 += 1.0 / rank
                break
    n = len(queries)
    print("\n" + "=" * 56)
    print("   EVALUATION RESULTS")
    print("=" * 56)
    print(f"  Queries     : {n}")
    print(f"  Hit Rate @3 : {100*hits3/n:.1f}%    (target >80%)")
    print(f"  MRR @5      : {mrr5/n:.4f}   (target >0.70)")
    print(f"  Avg Latency : {total_lat/n:.2f}s     (target <5s)")
    hit_ok = hits3/n > 0.8
    mrr_ok = mrr5/n > 0.7
    lat_ok = total_lat/n < 5.0
    print(f"  Status      : Hit={'✓' if hit_ok else '✗'}  MRR={'✓' if mrr_ok else '✗'}  Lat={'✓' if lat_ok else '✗'}")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()