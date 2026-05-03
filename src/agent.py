"""
src/agent.py — BIS Standards LLM Agent  [v4 - CONSUMER QUERY FIX]
==================================================================
Root cause of Q34/Q36 misses: stale disk cache stored wrong LLM answers
from before the retriever synonym map was fixed. Cache keys now include
a VERSION string — bumping it invalidates all old entries automatically.

Other fixes:
  - LLM system prompt now includes consumer→technical mapping guidance
  - SmartSkip threshold = 3.0x (very conservative, only skip obvious cases)
  - Retry on 429 with exponential backoff
  - Thread-safe cache and throttle locks
"""

import os, re, json, time, random, hashlib, logging, threading, urllib.request, urllib.error
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Cache version — bump this string to invalidate all cached responses ───────
# Change whenever retriever, prompt, or synonym map changes significantly.
CACHE_VERSION = "v4-consumer"

# ── Persistent disk cache ─────────────────────────────────────────────────────
_CACHE_FILE  = Path("data/llm_cache.json")
_llm_cache:  dict = {}
_cache_dirty = False
_cache_lock  = threading.Lock()

def _load_disk_cache():
    global _llm_cache
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _llm_cache = json.load(f)
            # Drop entries from old cache versions
            old_count = len(_llm_cache)
            _llm_cache = {k: v for k, v in _llm_cache.items()
                          if k.startswith(CACHE_VERSION + ":") or ":" not in k[:20]}
            dropped = old_count - len(_llm_cache)
            if dropped:
                log.info(f"[Cache] Dropped {dropped} stale entries (version bump)")
            log.info(f"[Cache] Loaded {len(_llm_cache)} cached responses")
        except Exception:
            _llm_cache = {}

def _save_disk_cache():
    global _cache_dirty
    if not _cache_dirty:
        return
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _cache_lock:
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_llm_cache, f, ensure_ascii=False)
        _cache_dirty = False
    except Exception as e:
        log.warning(f"[Cache] Save failed: {e}")

def _cache_key(query: str, backend: str) -> str:
    h = hashlib.md5(f"{backend}:{query}".encode()).hexdigest()
    return f"{CACHE_VERSION}:{h}"

def _cache_get(key: str) -> Optional[list]:
    with _cache_lock:
        return _llm_cache.get(key)

def _cache_set(key: str, value: list):
    global _cache_dirty
    with _cache_lock:
        _llm_cache[key] = value
        _cache_dirty = True
    if len(_llm_cache) % 5 == 0:
        _save_disk_cache()

_load_disk_cache()

# ── Thread-safe Groq throttle ─────────────────────────────────────────────────
_last_groq_call = 0.0
_groq_lock      = threading.Lock()
GROQ_MIN_INTERVAL = 2.1  # 60s / 30 req + 0.1s buffer

def _groq_throttle():
    global _last_groq_call
    with _groq_lock:
        elapsed = time.time() - _last_groq_call
        if elapsed < GROQ_MIN_INTERVAL:
            time.sleep(GROQ_MIN_INTERVAL - elapsed)
        _last_groq_call = time.time()

# ── System prompt — includes consumer query guidance ──────────────────────────
SYSTEM_PROMPT = """You are a Bureau of Indian Standards (BIS) expert on building material standards (SP 21).

Given a query and candidate IS standards, select the TOP 5 most relevant.

CONSUMER QUERY GUIDANCE — map informal language to technical standards:
- "house construction", "general purpose", "RCC", "plastering" → IS 8112 (43 Grade OPC)
- "high strength", "roof slab", "water tank", "bridges", "structural RCC" → IS 12269 (53 Grade OPC)
- "brickwork mortar", "smooth finish", "plastering", "masonry" → IS 1489 Part 1 (PPC fly ash)
- "repair work", "early strength", "quick set" → IS 8041 (Rapid Hardening)
- "rainy season", "water repellent", "damp areas" → IS 8043 (Hydrophobic) or IS 8112
- "tiles fixing", "tile adhesive" → IS 15477 (tile adhesive)
- "white cement", "decorative" → IS 8042 (White Portland Cement)
- "marine", "aggressive water", "sulphate" → IS 12330 or IS 6909
- "33 grade" → IS 269; "43 grade" → IS 8112; "53 grade" → IS 12269

Output ONLY a JSON array ordered by relevance:
[{"std_id": "IS XXX: YYYY", "rationale": "one sentence"}, ...]

Rules:
- Use ONLY std_ids from the candidates list — never invent IDs
- Return 3-5 items. No markdown, no text outside the JSON array."""


def _build_user_message(query: str, candidates: list) -> str:
    lines = []
    for i, c in enumerate(candidates[:8], 1):
        scope = c.get("scope", c.get("text_snippet", ""))[:200]
        lines.append(f"[{i}] {c['std_id']}: {c['title']}\n    {scope}")
    return f"Query: {query}\n\nCandidates:\n" + "\n".join(lines) + "\n\nReturn JSON array."


def _parse_llm_response(raw: str, valid_ids: set) -> Optional[list]:
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        reranked = json.loads(raw)
        reranked = [r for r in reranked if r.get("std_id") in valid_ids]
        return reranked if reranked else None
    except (json.JSONDecodeError, TypeError):
        return None


# ── Backend: Groq ─────────────────────────────────────────────────────────────

_GROQ_ALIASES = {
    "llama3-70b-8192":    "llama-3.3-70b-versatile",
    "llama3-8b-8192":     "llama-3.1-8b-instant",
    "mixtral-8x7b-32768": "llama-3.3-70b-versatile",
}

def _rerank_groq(query: str, candidates: list, api_key: str,
                 model: str = "llama-3.3-70b-versatile") -> Optional[list]:
    if not api_key:
        log.warning("[Groq] No GROQ_API_KEY — set env var or use --groq_key")
        return None

    model = _GROQ_ALIASES.get(model, model)
    ck    = _cache_key(query, f"groq:{model}")

    cached = _cache_get(ck)
    if cached is not None:
        log.info("[Groq] Cache hit — 0ms")
        return cached

    _groq_throttle()

    def _call():
        try:
            from groq import Groq
            t0   = time.time()
            resp = Groq(api_key=api_key).chat.completions.create(
                model       = model,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_message(query, candidates)},
                ],
                max_tokens  = 300,
                temperature = 0.0,
            )
            ms     = int((time.time() - t0) * 1000)
            raw    = resp.choices[0].message.content.strip()
            result = _parse_llm_response(raw, {c["std_id"] for c in candidates})
            if result:
                log.info(f"[Groq/{model}] {ms}ms → {[r['std_id'] for r in result]}")
                _cache_set(ck, result)
            else:
                log.warning(f"[Groq] Parse failed: {raw[:120]}")
            return result
        except ImportError:
            log.warning("[Groq] pip install groq")
            return None
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                return "RATE_LIMITED"
            log.warning(f"[Groq] {type(e).__name__}: {e}")
            return None

    result = _call()
    if result == "RATE_LIMITED":
        wait = 3.0 + random.uniform(0, 2.0)
        log.warning(f"[Groq] 429 — retrying in {wait:.1f}s")
        time.sleep(wait)
        result = _call()
        if result == "RATE_LIMITED":
            log.warning("[Groq] 429 again — falling back to retrieval order")
            return None
    return result


# ── Backend: Anthropic ────────────────────────────────────────────────────────

def _rerank_anthropic(query: str, candidates: list, api_key: str) -> Optional[list]:
    ck     = _cache_key(query, "anthropic")
    cached = _cache_get(ck)
    if cached is not None:
        return cached
    try:
        import anthropic
        resp   = anthropic.Anthropic(api_key=api_key).messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 300,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": _build_user_message(query, candidates)}],
        )
        result = _parse_llm_response(resp.content[0].text.strip(), {c["std_id"] for c in candidates})
        if result:
            _cache_set(ck, result)
        return result
    except Exception as e:
        log.warning(f"[Anthropic] {e}")
        return None


# ── Backend: Ollama ───────────────────────────────────────────────────────────

def _rerank_ollama(query: str, candidates: list,
                   model: str = "llama3",
                   host: str = "http://localhost:11434") -> Optional[list]:
    ck     = _cache_key(query, f"ollama:{model}")
    cached = _cache_get(ck)
    if cached is not None:
        return cached
    payload = json.dumps({
        "model": model, "stream": False,
        "prompt": SYSTEM_PROMPT + "\n\n" + _build_user_message(query, candidates),
        "options": {"temperature": 0.0},
    }).encode()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{host}/api/generate", data=payload,
                                   headers={"Content-Type": "application/json"}),
            timeout=120
        ) as r:
            raw = json.loads(r.read()).get("response", "")
        result = _parse_llm_response(raw, {c["std_id"] for c in candidates})
        if result:
            _cache_set(ck, result)
        return result
    except urllib.error.URLError:
        log.warning(f"[Ollama] Cannot connect to {host}. Run: ollama serve")
        return None
    except Exception as e:
        log.warning(f"[Ollama] {e}")
        return None


# ── Unified reranker ──────────────────────────────────────────────────────────

def rerank_with_llm(query, candidates, backend="groq", api_key=None,
                    ollama_model="llama3", groq_model="llama-3.3-70b-versatile"):
    if backend == "groq":
        return _rerank_groq(query, candidates,
                            api_key or os.environ.get("GROQ_API_KEY", ""), groq_model)
    elif backend == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        return _rerank_anthropic(query, candidates, key) if key else None
    elif backend == "ollama":
        return _rerank_ollama(query, candidates, ollama_model)
    return None


# ── BISAgent ──────────────────────────────────────────────────────────────────

class BISAgent:
    """
    SmartSkip at 3.0x: only skip LLM when retriever score is overwhelmingly
    dominant (ratio >= 3x). At 2x or lower, consumer queries like
    'cement for tiles fixing' can have ambiguous top scores and need LLM.
    """
    SMART_SKIP_RATIO = 3.0

    def __init__(self, retriever, api_key=None, use_llm=True,
                 llm_backend="groq", ollama_model="llama3",
                 groq_model="llama-3.3-70b-versatile",
                 model="claude-haiku-4-5"):   # legacy compat
        self.retriever    = retriever
        self.api_key      = api_key
        self.llm_backend  = llm_backend if use_llm else "none"
        self.ollama_model = ollama_model
        self.groq_model   = groq_model

        if self.llm_backend == "none":
            log.info("LLM: DISABLED")
        else:
            log.info(f"LLM: {self.llm_backend.upper()} / {self.groq_model}")
            if self.llm_backend == "groq":
                key = self.api_key or os.environ.get("GROQ_API_KEY", "")
                if key:
                    log.info(f"  Key: {key[:12]}...{key[-4:]}")
                else:
                    log.warning("  NO GROQ_API_KEY! → console.groq.com (free)")

    def answer(self, query: str, top_k: int = 5) -> list:
        candidates = self.retriever.retrieve(query, top_k=15)
        if not candidates:
            return []

        # SmartSkip — only when retriever is overwhelmingly dominant
        skip_llm = False
        if self.llm_backend != "none" and len(candidates) >= 2:
            t = candidates[0].get("score", 0)
            s = candidates[1].get("score", 0)
            if s > 0 and t >= self.SMART_SKIP_RATIO * s:
                skip_llm = True
                log.info(f"[SmartSkip] {t:.3f}/{s:.3f}={t/s:.1f}x ≥ {self.SMART_SKIP_RATIO}x")

        if self.llm_backend != "none" and not skip_llm:
            reranked = rerank_with_llm(
                query, candidates,
                backend=self.llm_backend, api_key=self.api_key,
                ollama_model=self.ollama_model, groq_model=self.groq_model,
            )
            if reranked:
                meta = {c["std_id"]: c for c in candidates}
                _save_disk_cache()
                return [
                    {
                        "std_id":    r["std_id"],
                        "title":     meta.get(r["std_id"], {}).get("title", ""),
                        "rationale": r.get("rationale", ""),
                        "score":     meta.get(r["std_id"], {}).get("score", 0),
                    }
                    for r in reranked[:top_k]
                ]

        # Fallback: retrieval order
        return [
            {
                "std_id":    c["std_id"],
                "title":     c["title"],
                "rationale": c.get("scope", c.get("text_snippet", ""))[:150],
                "score":     c["score"],
            }
            for c in candidates[:top_k]
        ]