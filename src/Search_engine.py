import json
import os
import pickle
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Run: pip install rank-bm25")

try:
    import faiss
except ImportError:
    raise ImportError("Run: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Run: pip install sentence-transformers")

CHUNKS_PATH = "chunks.json"  # output from Person 1
FAISS_INDEX_PATH = "faiss_index.bin"  # cached FAISS index
BM25_CACHE_PATH = "bm25_cache.pkl"  # cached BM25 object + corpus
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast & accurate 384-dim model
TOP_K_DEFAULT = 20

def _load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Make sure Person 1 has generated chunks.json."
        )
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[retrieval] Loaded {len(chunks)} chunks from '{path}'.")
    return chunks
def _tokenize(text: str) -> list[str]:
    return text.lower().split()

def _chunk_to_text(chunk: dict) -> str:
    parts = [
        chunk.get("is_number", ""),
        chunk.get("title", ""),
        chunk.get("content", ""),
    ]
    return " ".join(p for p in parts if p).strip()

class _BM25Index:

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        if os.path.exists(BM25_CACHE_PATH):
            print("[BM25] Loading index from cache …")
            with open(BM25_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            self._bm25 = cache["bm25"]
            self._corpus = cache["corpus"]  # raw texts (for verification)
        else:
            print("[BM25] Building index (first run) …")
            self._corpus = [_chunk_to_text(c) for c in chunks]
            tokenized = [_tokenize(text) for text in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
            with open(BM25_CACHE_PATH, "wb") as f:
                pickle.dump({"bm25": self._bm25, "corpus": self._corpus}, f)
            print(f"[BM25] Index cached to '{BM25_CACHE_PATH}'.")

    def search(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "is_number": chunk.get("is_number", ""),
                "title": chunk.get("title", ""),
                "content": chunk.get("content", ""),
                "category": chunk.get("category", ""),
            })
        return results

class _FAISSIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        dim = self._model.get_sentence_embedding_dimension()

        if os.path.exists(FAISS_INDEX_PATH):
            print("[FAISS] Loading index from cache …")
            self._index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            print("[FAISS] Building index — encoding chunks (may take a minute) …")
            texts = [_chunk_to_text(c) for c in chunks]
            embeddings = self._model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,  # cosine sim via inner-product
                convert_to_numpy=True,
            ).astype("float32")

            self._index = faiss.IndexFlatIP(dim)  # inner-product = cosine (after normalisation)
            self._index.add(embeddings)
            faiss.write_index(self._index, FAISS_INDEX_PATH)
            print(f"[FAISS] Index cached to '{FAISS_INDEX_PATH}'.")

    def search(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        q_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self._index.search(q_vec, top_k)
        scores, indices = scores[0], indices[0]  # unwrap batch dim

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx == -1:  # FAISS padding sentinel
                continue
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "is_number": chunk.get("is_number", ""),
                "title": chunk.get("title", ""),
                "content": chunk.get("content", ""),
                "category": chunk.get("category", ""),
            })
        return results
_chunks: list[dict] | None = None
_bm25_index: _BM25Index | None = None
_faiss_index: _FAISSIndex | None = None


def _ensure_loaded():
    global _chunks, _bm25_index, _faiss_index
    if _chunks is None:
        _chunks = _load_chunks()
    if _bm25_index is None:
        _bm25_index = _BM25Index(_chunks)
    if _faiss_index is None:
        _faiss_index = _FAISSIndex(_chunks)

def bm25_search(query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
    _ensure_loaded()
    return _bm25_index.search(query, top_k)

def faiss_search(query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
    _ensure_loaded()
    return _faiss_index.search(query, top_k)

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "ordinary portland cement"

    print(f"\n{'=' * 60}")
    print(f"Query: {query!r}")
    print(f"{'=' * 60}\n")

    print("── BM25 top 5 ──────────────────────────────────────────────")
    for r in bm25_search(query, top_k=5):
        print(f"  [{r['rank']}] score={r['score']:.4f}  {r['is_number']}  {r['title'][:60]}")

    print("\n── FAISS top 5 ─────────────────────────────────────────────")
    for r in faiss_search(query, top_k=5):
        print(f"  [{r['rank']}] score={r['score']:.4f}  {r['is_number']}  {r['title'][:60]}")
