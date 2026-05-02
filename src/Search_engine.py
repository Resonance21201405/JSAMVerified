from __future__ import annotations
import hashlib
import io
import json
import logging
import os
import pickle
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

# ── Project-root-relative paths ──────────────────────────────────────────────
# search_engine.py lives in  <root>/src/
# data files live in         <root>/data/
# cache files are written to <root>/data/   (same folder, keeps root clean)
_SRC_DIR:  Path = Path(__file__).resolve().parent       # .../src
_ROOT_DIR: Path = _SRC_DIR.parent                       # .../JSAMVerified
_DATA_DIR: Path = _ROOT_DIR / "data"                    # .../JSAMVerified/data

import numpy as np
import numpy.typing as npt

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise ImportError("Run: pip install rank-bm25") from e

try:
    import faiss
except ImportError as e:
    raise ImportError("Run: pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Run: pip install sentence-transformers") from e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log: logging.Logger = logging.getLogger("retrieval")


# ── Custom exceptions ────────────────────────────────────────────────────────

class RetrievalException(Exception):
    pass

class ChunksNotFoundError(RetrievalException):
    pass

class IndexBuildError(RetrievalException):
    pass

class SearchError(RetrievalException):
    pass


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Config:
    chunks_path:         str = str(_DATA_DIR / "chunks.json")
    faiss_index_path:    str = str(_DATA_DIR / "faiss_index.bin")
    bm25_cache_path:     str = str(_DATA_DIR / "bm25_cache.pkl")
    hash_cache_path:     str = str(_DATA_DIR / ".chunks_hash")
    embedding_model:     str = "all-MiniLM-L6-v2"
    top_k_default:       int = 15
    content_snippet_len: int = 200
    encode_batch_size:   int = 64
    cache_ttl_seconds:   int = 3600
    max_cache_size:      int = 1000
    num_threads:         int = 4
    device:              str = "cpu"

    stopwords: frozenset[str] = field(default_factory=lambda: frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "this", "that", "these", "those",
        "it", "its", "as", "into", "about", "which", "when", "where", "who",
        "how", "what",
    }))

    def validate(self) -> None:
        if self.top_k_default < 1:
            raise ValueError(f"top_k_default must be positive, got {self.top_k_default}")
        if self.content_snippet_len < 1:
            raise ValueError("content_snippet_len must be positive")
        if self.encode_batch_size < 1:
            raise ValueError("encode_batch_size must be positive")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        if self.max_cache_size < 1:
            raise ValueError("max_cache_size must be positive")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")


CFG = _Config()
CFG.validate()


# ── TTL Query Cache (thread-safe LRU + expiry) ───────────────────────────────

class TTLCache:
    def __init__(self, max_size: int = 128, ttl_seconds: int = 3600) -> None:
        self.max_size    = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = (value, time.time())
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


search_cache = TTLCache(max_size=CFG.max_cache_size, ttl_seconds=CFG.cache_ttl_seconds)


# ── Auto-invalidation: rebuild indexes when chunks.json changes ──────────────

def _md5_of_file(path: str) -> str:
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
        return h.hexdigest()
    except (IOError, OSError) as e:
        log.error(f"Failed to compute hash for {path}: {e}")
        raise

def _read_stored_hash() -> str:
    if not os.path.exists(CFG.hash_cache_path):
        return ""
    try:
        with open(CFG.hash_cache_path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except (IOError, OSError) as e:
        log.warning(f"Failed to read stored hash: {e}")
        return ""

def _write_stored_hash(digest: str) -> None:
    try:
        with open(CFG.hash_cache_path, "w", encoding="utf-8") as fh:
            fh.write(digest)
    except (IOError, OSError) as e:
        log.warning(f"Failed to write stored hash: {e}")

def _chunks_changed() -> bool:
    if not os.path.exists(CFG.chunks_path):
        return False
    try:
        current  = _md5_of_file(CFG.chunks_path)
        previous = _read_stored_hash()
        if current != previous:
            log.info("chunks.json changed — rebuilding all indexes.")
            for stale in (CFG.bm25_cache_path, CFG.faiss_index_path):
                if os.path.exists(stale):
                    try:
                        os.remove(stale)
                    except (IOError, OSError) as e:
                        log.warning(f"Failed to remove {stale}: {e}")
            search_cache.clear()
            _write_stored_hash(current)
            return True
        return False
    except Exception as e:
        log.error(f"Error checking chunks hash: {e}")
        return False


# ── Chunk loading ────────────────────────────────────────────────────────────

def _load_chunks() -> list[dict[str, Any]]:
    if not os.path.exists(CFG.chunks_path):
        raise ChunksNotFoundError(
            f"'{CFG.chunks_path}' not found. Run ingestion.py first."
        )
    try:
        with open(CFG.chunks_path, "r", encoding="utf-8") as fh:
            data: Any = json.load(fh)
    except json.JSONDecodeError as e:
        raise ChunksNotFoundError(f"Invalid JSON in {CFG.chunks_path}: {e}") from e
    except (IOError, OSError) as e:
        raise ChunksNotFoundError(f"Cannot read {CFG.chunks_path}: {e}") from e

    if not isinstance(data, list) or len(data) == 0:
        raise ChunksNotFoundError("chunks.json must be a non-empty JSON array.")

    required = {"is_number", "title", "content", "category"}
    for i, chunk in enumerate(data):
        if not isinstance(chunk, dict):
            raise ChunksNotFoundError(f"Chunk[{i}] is not a dict — got {type(chunk).__name__}")
        missing = required - chunk.keys()
        if missing:
            log.warning("Chunk[%d] missing keys: %s", i, missing)

    log.info("Loaded %d chunks from '%s'.", len(data), CFG.chunks_path)
    return data  # type: ignore[return-value]


# ── Text representation ──────────────────────────────────────────────────────

def _chunk_to_text(chunk: dict[str, Any]) -> str:
    """
    Build the searchable text for a chunk — used by BOTH BM25 and FAISS.

    Key improvements over the original advanced version:
      - is_number ×2       : boosts direct IS-number lookups in BM25
      - title     ×4       : strongest discriminative field; higher TF
                             helps BM25 rank it first  (was ×2)
      - scope     included : most concise, precise 1-sentence description —
                             best semantic signal for FAISS  (was missing)
      - sub_category       : helps disambiguate similar standards (was missing)

    ⚠️  After changing this function you MUST delete:
          bm25_cache.pkl   and   faiss_index.bin
        so both indexes rebuild with the new text representation.
        The MD5-based auto-invalidation handles this automatically if
        chunks.json itself changes, but NOT if only this function changes.
    """
    is_num   = chunk.get("is_number",    "")
    title    = chunk.get("title",        "")
    scope    = chunk.get("scope",        "")   # ← was missing
    sub_cat  = chunk.get("sub_category", "")   # ← was missing
    category = chunk.get("category",     "")
    content  = chunk.get("content",      "")

    return (
        f"{is_num} {is_num} "                  # ×2  (was ×1)
        f"{title} {title} {title} {title} "    # ×4  (was ×2)
        f"{scope} "
        f"{sub_cat} {category} "
        f"{content}"
    ).strip()


# ── Tokenisation ─────────────────────────────────────────────────────────────

def _validate_query(search_query: str, top_k: int) -> None:
    if not isinstance(search_query, str) or not search_query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")


@lru_cache(maxsize=256)
def _tokenize(text: str) -> tuple[str, ...]:
    """
    Custom tokeniser with IS-number normalisation and stopword removal.
    Results cached via lru_cache for repeated calls.
    """
    text   = re.sub(r"\bIS\s*(\d+)\s*:\s*(\d+)\b", r"IS\1:\2", text, flags=re.IGNORECASE)
    text   = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    return tuple(t for t in tokens if t not in CFG.stopwords or t.startswith("is"))


# ── Result helpers ───────────────────────────────────────────────────────────

def _content_snippet(content: str) -> str:
    content = content.strip()
    lim     = CFG.content_snippet_len
    return content[:lim] + "…" if len(content) > lim else content

def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]

def _build_result(
    rank: int, score: float, normalized_score: float,
    engine: str, chunk: dict[str, Any],
) -> dict[str, Any]:
    content: str = chunk.get("content", "")
    return {
        "rank":             rank,
        "score":            round(score, 6),
        "normalized_score": round(normalized_score, 6),
        "search_engine":    engine,
        "is_number":        chunk.get("is_number", ""),
        "title":            chunk.get("title", ""),
        "category":         chunk.get("category", ""),
        "content":          content,
        "content_snippet":  _content_snippet(content),
    }


# ── Weighted RRF candidate merging ───────────────────────────────────────────

def _build_combined_candidates(
    bm25_results:  list[dict[str, Any]],
    faiss_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge BM25 and FAISS result lists using weighted Reciprocal Rank Fusion.

    FIX vs original: the original sorted by avg_rank (equal weights).
    We weight FAISS ×2 vs BM25 ×1 to compensate for BM25's weakness on
    paraphrase/semantic queries (e.g. "aggregates for structural concrete").

    RRF formula per item:  score += weight / (k + rank)
    k = 60 is the standard smoothing constant.
    """
    K = 60
    rrf_scores: dict[str, float]         = {}
    meta:        dict[str, dict[str, Any]] = {}

    def _add(results: list[dict[str, Any]], weight: float) -> None:
        for item in results:
            key = item["is_number"]
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (K + item["rank"])
            meta[key] = item

    _add(bm25_results,  weight=1.0)   # BM25  counts once
    _add(faiss_results, weight=2.0)   # FAISS counts twice  ← weighted RRF

    bm25_map  = {r["is_number"]: r for r in bm25_results}
    faiss_map = {r["is_number"]: r for r in faiss_results}

    candidates: list[dict[str, Any]] = []
    for is_number, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        b      = bm25_map.get(is_number)
        f      = faiss_map.get(is_number)
        source = b or f
        assert source is not None
        found_in = (["bm25"] if b else []) + (["faiss"] if f else [])
        ranks    = [r for r in [b["rank"] if b else None, f["rank"] if f else None] if r]
        candidates.append({
            "is_number":        is_number,
            "title":            source["title"],
            "category":         source["category"],
            "content":          source["content"],
            "content_snippet":  source["content_snippet"],
            "found_in":         found_in,
            "found_in_count":   len(found_in),
            "bm25_rank":        b["rank"] if b else None,
            "faiss_rank":       f["rank"] if f else None,
            "bm25_norm_score":  b["normalized_score"] if b else None,
            "faiss_norm_score": f["normalized_score"] if f else None,
            "avg_rank":         round(sum(ranks) / len(ranks), 2),
            "rrf_score":        round(rrf_score, 6),
        })

    for i, c in enumerate(candidates, start=1):
        c["candidate_priority"] = i
    return candidates


# ── BM25 Index ───────────────────────────────────────────────────────────────

class _BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        try:
            if os.path.exists(CFG.bm25_cache_path):
                self._load_from_cache()
            else:
                self._build_index()
        except Exception as e:
            raise IndexBuildError(f"Failed to initialize BM25 index: {e}") from e

    def _load_from_cache(self) -> None:
        log.info("[BM25] Loading from cache …")
        t0 = time.perf_counter()
        try:
            with open(CFG.bm25_cache_path, "rb") as f:
                cache: dict[str, Any] = pickle.load(f)
            self._bm25:   BM25Okapi = cache["bm25"]
            self._corpus: list[str] = cache["corpus"]
            log.info("[BM25] Loaded in %.2fs.", time.perf_counter() - t0)
        except (pickle.PickleError, EOFError, KeyError) as e:
            log.warning(f"[BM25] Cache corrupt, rebuilding: {e}")
            if os.path.exists(CFG.bm25_cache_path):
                os.remove(CFG.bm25_cache_path)
            self._build_index()

    def _build_index(self) -> None:
        log.info("[BM25] Building index …")
        t0           = time.perf_counter()
        self._corpus = [_chunk_to_text(c) for c in self.chunks]
        tokenized    = [list(_tokenize(text)) for text in self._corpus]
        self._bm25   = BM25Okapi(tokenized)
        try:
            with open(CFG.bm25_cache_path, "wb") as f:
                pickle.dump({"bm25": self._bm25, "corpus": self._corpus}, f)
        except (IOError, OSError) as e:
            log.warning(f"[BM25] Failed to save cache: {e}")
        log.info("[BM25] Built in %.2fs.", time.perf_counter() - t0)

    def search(self, search_query: str, top_k: int = CFG.top_k_default) -> list[dict[str, Any]]:
        try:
            tokenized_q = list(_tokenize(search_query))
            if not tokenized_q:
                log.warning(f"Query '{search_query}' produced no tokens")
                return []
            scores      = self._bm25.get_scores(tokenized_q)
            ranked_idx  = np.argsort(scores)[::-1][:top_k].tolist()
            raw_scores  = [float(scores[i]) for i in ranked_idx]
            norm_scores = _normalize_scores(raw_scores)
            return [
                _build_result(rank, raw_scores[rank-1], norm_scores[rank-1], "bm25", self.chunks[idx])
                for rank, idx in enumerate(ranked_idx, start=1)
            ]
        except Exception as e:
            raise SearchError(f"BM25 search failed: {e}") from e


# ── FAISS Index ──────────────────────────────────────────────────────────────

class _FAISSIndex:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        try:
            self._model = SentenceTransformer(CFG.embedding_model, device=CFG.device)
            try:
                dim: int = self._model.get_embedding_dimension()
            except AttributeError:
                dim = self._model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]

            if os.path.exists(CFG.faiss_index_path):
                self._load_from_cache(dim)
            else:
                self._build_index(dim)
        except Exception as e:
            raise IndexBuildError(f"Failed to initialize FAISS index: {e}") from e

    def _load_from_cache(self, dim: int) -> None:
        log.info("[FAISS] Loading from cache …")
        t0 = time.perf_counter()
        try:
            self._index: faiss.Index = faiss.read_index(CFG.faiss_index_path)
            if self._index.d != dim:
                log.warning(
                    "[FAISS] Cached index dim=%d != model dim=%d — rebuilding.",
                    self._index.d, dim,
                )
                raise ValueError("dim mismatch")
            log.info("[FAISS] Loaded in %.2fs.", time.perf_counter() - t0)
        except Exception as e:
            log.warning(f"[FAISS] Cache corrupt/stale, rebuilding: {e}")
            if os.path.exists(CFG.faiss_index_path):
                os.remove(CFG.faiss_index_path)
            self._build_index(dim)

    def _build_index(self, dim: int) -> None:
        log.info("[FAISS] Building index — encoding %d chunks …", len(self.chunks))
        t0    = time.perf_counter()
        texts = [_chunk_to_text(c) for c in self.chunks]
        try:
            emb: npt.NDArray[np.float32] = self._model.encode(
                texts,
                batch_size=CFG.encode_batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(emb)
            faiss.write_index(self._index, CFG.faiss_index_path)
            log.info("[FAISS] Built in %.2fs.", time.perf_counter() - t0)
        except Exception as e:
            raise IndexBuildError(f"Failed to build FAISS index: {e}") from e

    def search(self, search_query: str, top_k: int = CFG.top_k_default) -> list[dict[str, Any]]:
        try:
            q_vec = self._model.encode(
                [search_query], normalize_embeddings=True, convert_to_numpy=True,
            ).astype("float32")
            raw_scores_np, raw_idx_np = self._index.search(q_vec, top_k)
            valid       = [(idx, sc) for idx, sc in zip(raw_idx_np[0], raw_scores_np[0]) if idx != -1]
            raw_scores  = [sc for _, sc in valid]
            norm_scores = _normalize_scores(raw_scores)
            return [
                _build_result(rank, raw_scores[rank-1], norm_scores[rank-1], "faiss", self.chunks[idx])
                for rank, (idx, _) in enumerate(valid, start=1)
            ]
        except Exception as e:
            raise SearchError(f"FAISS search failed: {e}") from e


# ── Singleton management (thread-safe) ──────────────────────────────────────

_lock:        threading.Lock               = threading.Lock()
_chunks:      list[dict[str, Any]] | None = None
_bm25_index:  _BM25Index          | None  = None
_faiss_index: _FAISSIndex         | None  = None


def _ensure_loaded() -> None:
    global _chunks, _bm25_index, _faiss_index
    with _lock:
        if _chunks is not None and _bm25_index is not None and _faiss_index is not None:
            return
        _chunks_changed()
        if _chunks is None:
            _chunks = _load_chunks()
        if _bm25_index is None:
            _bm25_index = _BM25Index(_chunks)
        if _faiss_index is None:
            _faiss_index = _FAISSIndex(_chunks)


# ── Public API ───────────────────────────────────────────────────────────────

def bm25_search(query: str, top_k: int = CFG.top_k_default) -> list[dict[str, Any]]:
    _validate_query(query, top_k)
    cache_key = f"bm25:{query}:{top_k}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        return cached
    _ensure_loaded()
    assert _bm25_index is not None
    result = _bm25_index.search(search_query=query, top_k=top_k)
    search_cache.set(cache_key, result)
    return result


def faiss_search(query: str, top_k: int = CFG.top_k_default) -> list[dict[str, Any]]:
    _validate_query(query, top_k)
    cache_key = f"faiss:{query}:{top_k}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        return cached
    _ensure_loaded()
    assert _faiss_index is not None
    result = _faiss_index.search(search_query=query, top_k=top_k)
    search_cache.set(cache_key, result)
    return result


def get_candidates(query: str, top_k: int = CFG.top_k_default) -> dict[str, Any]:
    """Return merged BM25 + FAISS candidates ranked by weighted RRF."""
    _validate_query(query, top_k)
    cache_key = f"candidates:{query}:{top_k}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        return cached
    b_results = bm25_search(query,  top_k=top_k)
    f_results = faiss_search(query, top_k=top_k)
    combined  = _build_combined_candidates(b_results, f_results)
    result    = {
        "query":               query,
        "top_k_per_engine":    top_k,
        "bm25_results":        b_results,
        "faiss_results":       f_results,
        "combined_candidates": combined,
    }
    search_cache.set(cache_key, result)
    return result


def batch_search(
    queries: list[str],
    engine:  Literal["bm25", "faiss", "both"] = "both",
    top_k:   int = CFG.top_k_default,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Run multiple queries in parallel using a thread pool."""
    if not queries:
        raise ValueError("queries list must not be empty.")

    def _search_one(q: str) -> tuple[str, dict[str, list[dict[str, Any]]]]:
        entry: dict[str, list[dict[str, Any]]] = {}
        if engine in ("bm25",  "both"): entry["bm25"]  = bm25_search(q,  top_k=top_k)
        if engine in ("faiss", "both"): entry["faiss"] = faiss_search(q, top_k=top_k)
        return q, entry

    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    with ThreadPoolExecutor(max_workers=CFG.num_threads) as ex:
        for q, entry in ex.map(_search_one, queries):
            results[q] = entry
    return results


def get_index_stats() -> dict[str, Any]:
    _ensure_loaded()
    assert _faiss_index is not None
    assert _chunks      is not None
    try:
        dim = _faiss_index._model.get_embedding_dimension()
    except AttributeError:
        dim = _faiss_index._model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]
    return {
        "num_chunks":         len(_chunks),
        "embedding_model":    CFG.embedding_model,
        "embedding_dim":      dim,
        "faiss_index_type":   type(_faiss_index._index).__name__,
        "bm25_cache_exists":  os.path.exists(CFG.bm25_cache_path),
        "faiss_cache_exists": os.path.exists(CFG.faiss_index_path),
        "chunks_hash":        _read_stored_hash() or "not computed",
        "cache_size":         search_cache.size(),
        "cache_ttl":          CFG.cache_ttl_seconds,
        "top_k_default":      CFG.top_k_default,
    }


def reset_indexes() -> None:
    """Delete all cached indexes; next search rebuilds from scratch."""
    global _chunks, _bm25_index, _faiss_index
    with _lock:
        for path in (CFG.bm25_cache_path, CFG.faiss_index_path, CFG.hash_cache_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                    log.info("Removed: %s", path)
                except (IOError, OSError) as e:
                    log.warning(f"Failed to remove {path}: {e}")
        _chunks = _bm25_index = _faiss_index = None
        search_cache.clear()
    log.info("All indexes reset — next search will rebuild from scratch.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="retrieval",
        description="IS-standards retrieval engine — BM25 + FAISS + weighted RRF",
    )
    parser.add_argument("query", nargs="?", default="ordinary portland cement")
    parser.add_argument("--engine", "-e",
        choices=["bm25", "faiss", "both", "candidates"], default="candidates")
    parser.add_argument("--top-k", "-k", type=int, default=CFG.top_k_default)
    parser.add_argument("--stats",  action="store_true")
    parser.add_argument("--reset",  action="store_true")
    args = parser.parse_args()

    try:
        if args.reset:
            reset_indexes(); print("✓ Indexes cleared."); return

        if args.stats:
            stats = get_index_stats()
            print("\n── Index Statistics ─────────────────────────────────")
            for k, v in stats.items():
                print(f"  {k:<26} {v}")
            return

        if args.engine == "candidates":
            data = get_candidates(args.query, top_k=args.top_k)
            print(f"\n  Query : {args.query!r}")
            print(f"  {'Pri':>3}  {'IS Number':<20}  {'Both':^4}  "
                  f"{'BM25':>4}  {'FAISS':>5}  {'RRF':>8}  Title")
            print(f"  {'─'*72}")
            for c in data["combined_candidates"]:
                both = "✓✓" if c["found_in_count"] == 2 else "  "
                br   = str(c["bm25_rank"])  if c["bm25_rank"]  is not None else "—"
                fr   = str(c["faiss_rank"]) if c["faiss_rank"] is not None else "—"
                print(f"  [{c['candidate_priority']:>2}]  {c['is_number']:<20}  {both:^4}  "
                      f"{br:>4}  {fr:>5}  {c['rrf_score']:>8.5f}  {c['title'][:32]}")
            return

        if args.engine in ("bm25", "both"):
            print("\n── BM25 Results ──")
            for r in bm25_search(args.query, top_k=args.top_k):
                print(f"  [{r['rank']:>2}] norm={r['normalized_score']:.3f}  "
                      f"{r['is_number']:<18}  {r['title'][:50]}")

        if args.engine in ("faiss", "both"):
            print("\n── FAISS Results ──")
            for r in faiss_search(args.query, top_k=args.top_k):
                print(f"  [{r['rank']:>2}] norm={r['normalized_score']:.3f}  "
                      f"{r['is_number']:<18}  {r['title'][:50]}")

    except (ChunksNotFoundError, IndexBuildError, SearchError, ValueError) as e:
        log.error(f"Error: {e}")
        print(f"❌ {e}")
        exit(1)


if __name__ == "__main__":
    _cli()