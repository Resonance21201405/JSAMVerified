import argparse
import time
from pathlib import Path
import sys

# Fix import path
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.append(str(SRC))

from src.search_engine import bm25_search, faiss_search, _ensure_loaded
from src.utils import (
    load_json, save_json,
    clean_query, expand_query,
    rrf_fuse, deduplicate_results,
    validate_against_chunks,
    normalise_is_number, CHUNKS_PATH,
)

# ── Cross-encoder (optional, disable for speed) ───────────────────────────
USE_CROSS_ENCODER = False

if USE_CROSS_ENCODER:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, candidates):
    if not USE_CROSS_ENCODER:
        return candidates

    pairs  = [(query, c["title"] + " " + c["content"][:300]) for c in candidates]
    scores = ce.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def pipeline(query, chunks):
    """
    Retrieval pipeline with weighted RRF.

    Key fixes:
    - BM25  receives cleaned + expanded query (extra synonym tokens boost recall).
    - FAISS receives the ORIGINAL natural-language query (sentence embeddings work
      best on natural text; cleaning/expansion degrades semantic similarity).
    - top_k raised to 20 for a wider candidate pool before RRF fusion.
    - FAISS counted twice in RRF (×2 semantic weight) to favour meaning over keywords.
    """
    q_bm25  = expand_query(clean_query(query))  # cleaned + expanded → BM25
    q_faiss = query.strip()                      # original natural text → FAISS

    bm25_results  = bm25_search(q_bm25,  top_k=20)
    faiss_results = faiss_search(q_faiss, top_k=20)

    # Weighted RRF: FAISS counted twice → semantic weight = 2× BM25
    fused    = rrf_fuse(bm25_results, faiss_results, faiss_results, top_n=10)

    reranked = rerank(query, fused)
    dedup    = deduplicate_results(reranked)
    valid    = validate_against_chunks(dedup, chunks)

    # Flat list of normalised IS numbers, top-5
    return [normalise_is_number(r["is_number"]) for r in valid[:5]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queries = load_json(args.input)
    chunks  = load_json(CHUNKS_PATH)

    # ── Warm-up: load ALL indexes before the query loop ────────────────────
    # Without this the first query pays cold-start cost (model load + index IO)
    print("[INFO] Warming up indexes (BM25 + FAISS + embedder)…")
    t_warm = time.time()
    _ensure_loaded()
    bm25_search("warm up",  top_k=1)
    faiss_search("warm up", top_k=1)
    print(f"[INFO] Warm-up done in {time.time() - t_warm:.2f}s\n")

    output = []
    start  = time.time()
    print(f"[INFO] Running {len(queries)} queries…\n")

    for i, q in enumerate(queries):
        t0 = time.time()

        retrieved = pipeline(q["query"], chunks)
        latency   = round(time.time() - t0, 2)

        # Accept both "id" (public_test_set) and "query_id" (transformed) key names
        record_id = q.get("id") or q.get("query_id") or str(i + 1)
        # Accept both "expected_standards" and "relevant" key names
        expected  = q.get("expected_standards") or q.get("relevant") or []

        output.append({
            "id":                  record_id,
            "query":               q["query"],
            "expected_standards":  expected,
            "retrieved_standards": retrieved,
            "latency_seconds":     latency,
        })

        print(f"[{i+1}/{len(queries)}] {q['query'][:70]} → {latency:.2f}s")

    total_time = round(time.time() - start, 2)
    avg        = round(total_time / max(len(queries), 1), 2)
    print(f"\nTotal: {total_time:.2f}s  |  Avg/query: {avg:.2f}s")

    save_json(output, args.output)
    print(f"[DONE] Saved → {args.output}")


if __name__ == "__main__":
    main()