"""
inference.py — MANDATORY entry point for the hackathon judges.

Usage:
    python inference.py --input input.json --output output.json

input.json  format (list of queries):
    [
      {"query_id": "q1", "query": "ordinary portland cement 33 grade"},
      {"query_id": "q2", "query": "fly ash cement specifications"},
      ...
    ]

output.json format (list of predictions):
    [
      {
        "query_id": "q1",
        "query":    "ordinary portland cement 33 grade",
        "results": [
          {"rank": 1, "is_number": "IS 269: 1989", "title": "...", "score": 0.0421, "category": "Cement"},
          ...
        ]
      },
      ...
    ]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── project imports ────────────────────────────────────────────────────────
from utils import (
    load_json, save_json,
    clean_query, expand_query,
    rrf_fuse, deduplicate_results, validate_against_chunks,
    normalise_is_number, CHUNKS_PATH, RESULTS_DIR,
)

# ── search engine (Person 2's deliverable) ────────────────────────────────
try:
    from Search_engine import bm25_search, faiss_search
    _SEARCH_ENGINE_AVAILABLE = True
except ImportError as e:
    _SEARCH_ENGINE_AVAILABLE = False
    print(f"[inference] Search engine unavailable ({e}).")


# ── optional cross-encoder (Person 3 dependency) ──────────────────────────
try:
    from sentence_transformers import CrossEncoder
    _CE_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    _CROSS_ENCODER_AVAILABLE = True
    print("[inference] Cross-encoder loaded ✓")
except Exception as e:
    _CROSS_ENCODER_AVAILABLE = False
    print(f"[inference] Cross-encoder unavailable ({e}). Falling back to RRF scores.")


# ── constants ──────────────────────────────────────────────────────────────
BM25_TOP_K      = 20   # candidates from BM25
FAISS_TOP_K     = 20   # candidates from FAISS
RRF_TOP_N       = 10   # after fusion
RERANK_TOP_N    = 10   # fed into cross-encoder
FINAL_TOP_N     = 5    # returned per query


# ──────────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────────

def _cross_encoder_rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Rerank *candidates* with the cross-encoder and return top_n."""
    if not _CROSS_ENCODER_AVAILABLE or not candidates:
        return candidates[:top_n]

    pairs = [(query, c.get("title", "") + " " + c.get("content", "")[:512])
             for c in candidates]
    ce_scores = _CE_MODEL.predict(pairs)

    for cand, ce_score in zip(candidates, ce_scores):
        cand["score"] = float(ce_score)

    reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return reranked[:top_n]


def run_pipeline(query: str, chunks: list[dict]) -> list[dict]:
    """
    Full retrieval pipeline for a single query.
    Returns a list of result dicts, ranked best-first.
    """
    # Step 1: query processing
    cleaned  = clean_query(query)
    expanded = expand_query(cleaned)

    # Step 2: dual retrieval
    bm25_results  = bm25_search(expanded,  top_k=BM25_TOP_K)
    faiss_results = faiss_search(expanded, top_k=FAISS_TOP_K)

    # Step 3: RRF fusion
    fused = rrf_fuse(bm25_results, faiss_results, k=60, top_n=RRF_TOP_N)

    # Step 4: cross-encoder rerank
    reranked = _cross_encoder_rerank(query, fused[:RERANK_TOP_N], top_n=FINAL_TOP_N)

    # Step 5: post-processing
    deduped   = deduplicate_results(reranked)
    validated = validate_against_chunks(deduped, chunks)

    # Normalise IS numbers and add final ranks
    final = []
    for rank, item in enumerate(validated[:FINAL_TOP_N], start=1):
        final.append({
            "rank":      rank,
            "is_number": normalise_is_number(item.get("is_number", "")),
            "title":     item.get("title", ""),
            "category":  item.get("category", ""),
            "score":     round(item.get("score", 0.0), 6),
        })

    return final


# ──────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IS-standard retrieval — inference entry point"
    )
    p.add_argument("--input",  required=True, help="Path to input JSON file")
    p.add_argument("--output", required=True, help="Path to write output JSON file")
    p.add_argument(
        "--top-n", type=int, default=FINAL_TOP_N,
        help=f"Number of results per query (default: {FINAL_TOP_N})"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-query results to stdout"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── load input ──────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    queries = load_json(input_path)
    if isinstance(queries, dict):          # allow single-query dicts too
        queries = [queries]

    # ── load chunks for validation ──────────────────────────────────────────
    try:
        chunks = load_json(CHUNKS_PATH)
    except FileNotFoundError:
        print(
            f"[ERROR] chunks.json not found at {CHUNKS_PATH}. "
            "Run src/ingestion.py first (Person 1 deliverable).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── run pipeline ────────────────────────────────────────────────────────
    output  = []
    t_start = time.perf_counter()

    print(f"[inference] Processing {len(queries)} queries …")
    for i, item in enumerate(queries, start=1):
        query    = item.get("query", "").strip()
        query_id = item.get("query_id", str(i))

        if not query:
            print(f"  [!] Query {query_id} is empty — skipping.")
            continue

        t0      = time.perf_counter()
        results = run_pipeline(query, chunks)
        elapsed = time.perf_counter() - t0

        output.append({
            "query_id": query_id,
            "query":    query,
            "results":  results,
        })

        if args.verbose:
            print(f"\n── [{i}/{len(queries)}] {query_id}: {query!r}  ({elapsed:.2f}s)")
            for r in results:
                print(f"   [{r['rank']}] {r['is_number']}  |  {r['title'][:60]}")
        else:
            bar = "#" * i + "." * (len(queries) - i)
            print(f"\r  [{bar}] {i}/{len(queries)}  {elapsed:.2f}s/q", end="", flush=True)

    total_time = time.perf_counter() - t_start
    print(f"\n[inference] Done — {len(output)} queries in {total_time:.1f}s "
          f"({total_time/max(len(output),1):.2f}s avg)")

    # ── save output ─────────────────────────────────────────────────────────
    output_path = Path(args.output)
    save_json(output, output_path)
    print(f"[inference] Results written → {output_path}")


if __name__ == "__main__":
    main()