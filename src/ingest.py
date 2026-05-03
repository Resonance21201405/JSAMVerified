"""
src/ingest.py — Build the retrieval index from chunks.json (already parsed).

This is a lightweight wrapper. The full PDF parsing is in ingestion.py at root.
This file just builds the TF-IDF sparse index from an existing chunks.json.

Called by inference.py when sparse_index.pkl is missing.
"""

import os
import json
import pickle
import logging

log = logging.getLogger(__name__)


def build_index(pdf_path: str, index_dir: str):
    """
    Build the retrieval index.

    If chunks.json already exists in index_dir, just rebuilds the sparse index.
    Otherwise, runs the full PDF ingestion pipeline first.
    """
    os.makedirs(index_dir, exist_ok=True)
    chunks_path = os.path.join(index_dir, "chunks.json")

    # ── Step 1: Get chunks ────────────────────────────────────────────────
    if not os.path.exists(chunks_path):
        log.info("chunks.json not found — running PDF ingestion...")
        _ingest_pdf(pdf_path, chunks_path)
    else:
        log.info(f"Using existing chunks.json ({chunks_path})")

    # ── Step 2: Build sparse TF-IDF index ────────────────────────────────
    sparse_path = os.path.join(index_dir, "sparse_index.pkl")
    if not os.path.exists(sparse_path):
        log.info("Building sparse TF-IDF index...")
        _build_sparse_index(chunks_path, sparse_path)
    else:
        log.info("sparse_index.pkl already exists, skipping")

    log.info("Index ready.")


def _ingest_pdf(pdf_path: str, output_chunks_path: str):
    """Run full PDF ingestion using the root ingestion.py logic."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from ingestion import parse_pdf, deduplicate, validate
        from pathlib import Path
        import json

        chunks = parse_pdf(Path(pdf_path))
        chunks = deduplicate(chunks)
        validate(chunks)

        os.makedirs(os.path.dirname(output_chunks_path), exist_ok=True)
        with open(output_chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {len(chunks)} chunks to {output_chunks_path}")

    except Exception as e:
        raise RuntimeError(f"PDF ingestion failed: {e}") from e


def _build_sparse_index(chunks_path: str, sparse_path: str):
    """Build TF-IDF vectorizer + matrix from chunks and save."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    def chunk_to_text(c: dict) -> str:
        # Weight: title ×3, scope ×2, content ×1  (aligned with retriever)
        title   = c.get("title", "")
        scope   = c.get("scope", "")
        cat     = c.get("category", "")
        sub_cat = c.get("sub_category", "")
        content = c.get("content", c.get("text", ""))
        return f"{title} {title} {title} {scope} {scope} {cat} {sub_cat} {content}".strip()

    texts = [chunk_to_text(c) for c in chunks]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),       # aligned with retriever (was (1,2))
        max_features=120_000,     # aligned with retriever (was 50000)
        sublinear_tf=True,
        min_df=1,
        max_df=0.92,              # added: suppress boilerplate terms
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform(texts)
    matrix = normalize(matrix)    # L2-normalise for cosine consistency

    # Save as sparse_merged.pkl so the retriever recognises it without rebuild
    out_path = sparse_path.replace("sparse_index.pkl", "sparse_merged.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)

    log.info(f"Sparse index: {matrix.shape[0]} docs × {matrix.shape[1]} features → {out_path}")