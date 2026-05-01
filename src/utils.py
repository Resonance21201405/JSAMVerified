"""
utils.py — shared helpers for the IS-standard retrieval project.

Used by: search_engine.py, reranker.py, inference.py, evaluation.py
"""

import json
import re
import string
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths (all modules import from here — change once, fix everywhere)
# ---------------------------------------------------------------------------
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
CHUNKS_PATH     = DATA_DIR / "chunks.json"
FAISS_INDEX_PATH= DATA_DIR / "faiss_index.bin"
BM25_CACHE_PATH = DATA_DIR / "bm25_cache.pkl"
RESULTS_DIR     = ROOT_DIR / "results"

# Make sure output dirs exist at import time
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# IS-number helpers
# ---------------------------------------------------------------------------

# Matches:  IS 269: 1989 | IS 1489 (Part 2): 1991 | IS 12269:2013  etc.
_IS_RE = re.compile(
    r"IS\s*(\d+(?:\s*\(\s*Part\s*\d+\s*\))?)\s*[:\-]?\s*(\d{4})",
    re.IGNORECASE,
)


def normalise_is_number(raw: str) -> str:
    """
    Convert any IS-number variant to the canonical form used in chunks.json.
    E.g. "is 269:1989" → "IS 269: 1989"
         "IS1489(Part2):1991" → "IS 1489 (Part 2): 1991"
    Returns the normalised string, or the original stripped string if no match.
    """
    raw = raw.strip()
    m = _IS_RE.search(raw)
    if not m:
        return raw
    number_part = m.group(1).strip()
    # Normalise Part annotation spacing
    number_part = re.sub(r"\s*\(\s*Part\s*(\d+)\s*\)\s*", r" (Part \1)", number_part, flags=re.IGNORECASE).strip()
    year = m.group(2)
    return f"IS {number_part}: {year}"


def extract_is_numbers(text: str) -> list[str]:
    """
    Extract all IS numbers from a free-text string and return them
    normalised.
    """
    return [normalise_is_number(m.group(0)) for m in _IS_RE.finditer(text)]


def is_numbers_equal(a: str, b: str) -> bool:
    """Case-insensitive, whitespace-insensitive IS-number comparison."""
    def _clean(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().upper())
    return _clean(normalise_is_number(a)) == _clean(normalise_is_number(b))


# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset("""
a an the and or but in on at to for of with by from is are was were
be been being have has had do does did will would shall should may
might must can could this that these those it its they their them
we our us i my me he she his her him
""".split())


def clean_query(query: str) -> str:
    """
    Light cleaning for a retrieval query:
    lowercase, strip punctuation, collapse whitespace.
    Does NOT remove stop-words so BM25 scoring stays accurate.
    """
    query = query.lower().strip()
    query = query.translate(str.maketrans("", "", string.punctuation))
    query = re.sub(r"\s+", " ", query)
    return query


def tokenize(text: str) -> list[str]:
    """Whitespace tokeniser (consistent with search_engine.py)."""
    return text.lower().split()


def expand_query(query: str) -> str:
    """
    Very lightweight query expansion — adds synonyms relevant to IS standards
    without touching the original tokens.
    Safe to call on every query; adds ≤6 extra tokens.
    """
    expansions: dict[str, list[str]] = {
        "cement":      ["binder", "mortar", "concrete"],
        "steel":       ["iron", "metal", "rebar", "bars"],
        "pipe":        ["tube", "conduit", "pipeline"],
        "aggregate":   ["sand", "gravel", "stone", "coarse", "fine"],
        "brick":       ["masonry", "clay", "block"],
        "water":       ["potable", "drinking", "supply"],
        "concrete":    ["rcc", "pcc", "mix", "grade"],
        "wire":        ["cable", "strand", "rope"],
        "paint":       ["coating", "primer", "enamel"],
        "timber":      ["wood", "lumber", "plywood"],
    }
    tokens = query.lower().split()
    extra: list[str] = []
    for token in tokens:
        if token in expansions:
            extra.extend(expansions[token])
    if extra:
        return query + " " + " ".join(extra)
    return query


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    print(f"[utils] Saved → {path}")


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_result(result: dict, rank: int | None = None) -> str:
    """
    Pretty-print a single ranked result dict for CLI output.
    """
    prefix = f"[{rank}] " if rank is not None else ""
    lines = [
        f"{prefix}{result.get('is_number', '?')}  —  {result.get('title', '')[:70]}",
        f"     Category : {result.get('category', 'N/A')}",
        f"     Score    : {result.get('score', 0.0):.4f}",
    ]
    if result.get("explanation"):
        lines.append(f"     Rationale: {result['explanation'][:120]} …")
    return "\n".join(lines)


def deduplicate_results(results: list[dict]) -> list[dict]:
    """
    Remove duplicate IS numbers from a ranked list, keeping highest rank.
    Preserves order.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for r in results:
        key = normalise_is_number(r.get("is_number", ""))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def validate_against_chunks(
    results: list[dict], chunks: list[dict]
) -> list[dict]:
    """
    Filter out any results whose IS number is not present in chunks.json.
    Guards against hallucination if LLM is ever used upstream.
    """
    valid_numbers = {normalise_is_number(c["is_number"]) for c in chunks}
    return [
        r for r in results
        if normalise_is_number(r.get("is_number", "")) in valid_numbers
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (used by inference.py + reranker.py)
# ---------------------------------------------------------------------------

def rrf_fuse(
    *ranked_lists: list[dict],
    k: int = 60,
    top_n: int = 10,
) -> list[dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        *ranked_lists : each list is ordered best-first, items must have
                        an 'is_number' key.
        k             : RRF smoothing constant (default 60 is standard).
        top_n         : how many fused results to return.

    Returns:
        List of dicts with keys: is_number, title, content, category, score.
        Sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    meta:   dict[str, dict]  = {}   # store last-seen metadata per IS number

    for ranked in ranked_lists:
        for rank_0based, item in enumerate(ranked, start=1):
            key = normalise_is_number(item.get("is_number", ""))
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank_0based)
            meta[key] = item  # keep latest metadata copy

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for is_num, score in fused:
        entry = dict(meta[is_num])  # copy metadata
        entry["is_number"] = is_num
        entry["score"] = round(score, 6)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # IS number normalisation
    tests = [
        ("is 269:1989",              "IS 269: 1989"),
        ("IS1489(Part2):1991",       "IS 1489 (Part 2): 1991"),
        ("IS 12269 : 2013",          "IS 12269: 2013"),
        ("IS 383-1970",              "IS 383: 1970"),
    ]
    print("=== IS normalisation tests ===")
    for raw, expected in tests:
        got = normalise_is_number(raw)
        status = "✓" if got == expected else f"✗  expected: {expected}"
        print(f"  {status}   '{raw}'  →  '{got}'")

    # RRF fusion
    list_a = [
        {"is_number": "IS 269: 1989",  "title": "OPC 33 Grade", "content": "", "category": "Cement", "score": 0.9},
        {"is_number": "IS 383: 1970",  "title": "Aggregates",   "content": "", "category": "Concrete", "score": 0.7},
    ]
    list_b = [
        {"is_number": "IS 383: 1970",  "title": "Aggregates",   "content": "", "category": "Concrete", "score": 0.95},
        {"is_number": "IS 8112: 1989", "title": "OPC 43 Grade", "content": "", "category": "Cement", "score": 0.6},
    ]
    fused = rrf_fuse(list_a, list_b, k=60, top_n=3)
    print("\n=== RRF fusion test ===")
    for r in fused:
        print(f"  {r['is_number']}  score={r['score']:.6f}")

    # Query expansion
    print("\n=== Query expansion test ===")
    q = "portland cement for concrete mix"
    print(f"  original : {q}")
    print(f"  expanded : {expand_query(q)}")