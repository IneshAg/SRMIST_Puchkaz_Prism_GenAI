"""
Stage 3 — Semantic Cache (Member A)

Serves a previously-computed, already-validated API response for a query
that is semantically the same as one seen before, without re-running
Stage 1 (LLM structuring) or Stage 2 (deeplink mapping) — this is what
gets a repeat/paraphrased "my screen is black" under the 300ms budget
instead of the multi-second Stage 1+2 pipeline.

Semantic cache key
-------------------
The key is NOT the raw query string (an exact-string cache would miss
"screen is black" vs "display won't turn on" vs a typo'd version of
either — exactly what Stage 0's query_variations exist to cover).
Instead:

  * every canonical query + its variations is embedded (embeddings.py)
    and every embedding is L2-normalized.
  * matching itself is done with one vectorized sparse dot product across
    every stored (normalized) vector at once — because vectors are
    pre-normalized, that dot product IS cosine similarity, so this scales
    to thousands of entries without a per-entry Python loop (the thing
    that actually blew the latency budget in the first version of this
    module: recomputing norms for the full cache on every lookup). See
    `get()`.

Low-confidence matches return None, not a guess — a near-miss served as
a hit would be a hallucinated answer to a question that wasn't actually
asked, which is exactly what the "no hallucinations" / "null instead of
low-confidence" requirement rules out.

Device gate
-----------
Char n-gram similarity between "Galaxy S22 screen is black" and
"Galaxy S24 screen is black" comes out ~0.9 — same symptom text dominates
the vector, the couple of digits that differ barely move it. But the
*device* is exactly the thing that can change which Settings deeplink or
step sequence is correct (foldable-only actions, tablet vs phone menus,
model-specific screens). So the cache never serves a hit across devices:
`device` is stored per vector and masked out before similarity ranking,
not left to the embedding to sort out.
"""
from __future__ import annotations

import copy
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import sparse

from embeddings import Embedder

DEFAULT_SIMILARITY_THRESHOLD = 0.80
_SIMHASH_BITS = 16
UNKNOWN_DEVICE = "unknown"


def _normalize_device(device: Optional[str]) -> str:
    """Canonical device key, so the same phone written differently still
    matches: "Galaxy Z Flip 7" / "Z Flip 7" / "Galaxy Flip7" -> "flip7",
    "Samsung Galaxy S24 Ultra" -> "s24ultra". Different models stay distinct."""
    d = (device or UNKNOWN_DEVICE).strip().lower()
    if d in ("", UNKNOWN_DEVICE, "samsung device", "techcorp device"):
        return UNKNOWN_DEVICE
    d = re.sub(r"\b(samsung|galaxy|techcorp|nexa)\b", " ", d)
    d = re.sub(r"\bz\s*(?=(flip|fold))", " ", d)
    d = re.sub(r"[\s\-_]+", "", d)
    return d or UNKNOWN_DEVICE


def _devices_compatible(a: str, b: str) -> bool:
    """Exact key match, "unknown" wildcard, or overlap of slash-alternatives
    ("a15/a16" covers a query for "a16")."""
    if a == b or a == UNKNOWN_DEVICE or b == UNKNOWN_DEVICE:
        return True
    return bool(set(a.split("/")) & set(b.split("/")))


@dataclass
class CacheEntry:
    canonical_query: str
    query_variations: List[str]
    response: dict
    device: str = UNKNOWN_DEVICE
    # Fingerprint of the siis_response this answer was extracted from (None =
    # unknown / not recorded). See get(context_key=...).
    context_key: Optional[str] = None
    hits: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class CacheResult:
    response: dict
    similarity: float
    matched_query: str
    latency_ms: float


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_entries: int = 5000,
        simhash_bits: int = _SIMHASH_BITS,
        seed: int = 42,
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

        self._entries: List[CacheEntry] = []
        # Row-parallel index over every (canonical + variation) vector ever
        # stored: row i belongs to entry `_row_entry[i]` and device
        # `_row_device[i]`. `_matrix` rows are L2-normalized, so
        # matrix @ query.T is cosine similarity directly.
        self._matrix: Optional[sparse.csr_matrix] = None
        self._row_entry: List[int] = []
        self._row_device: List[str] = []
        self._row_context: List[Optional[str]] = []
        self._pending: List[sparse.csr_matrix] = []  # batched, flushed lazily

        self._simhash_planes: Optional[np.ndarray] = None
        self._simhash_bits = simhash_bits
        self._rng = np.random.default_rng(seed)

        # Guards every read/write below. Fine for the hackathon's single-process
        # demo/eval harness either way, but get()/put() each do several sequential
        # mutations (_flush_pending appending to _matrix, _evict_oldest rebuilding
        # it, entry.hits += 1) that are not atomic on their own -- a lock keeps two
        # concurrent requests from ever observing or leaving the index half-updated.
        self._lock = threading.Lock()

    # -- semantic key -----------------------------------------------------

    def _ensure_planes(self, dim: int) -> None:
        if self._simhash_planes is None:
            self._simhash_planes = self._rng.standard_normal((self._simhash_bits, dim)).astype(np.float32)

    def semantic_cache_key(self, vec) -> str:
        """Random-hyperplane SimHash: embedding -> short bucket key.
        Vectors on the same side of all hyperplanes get the same key, so
        near-duplicate embeddings collide into the same bucket. Used to
        stamp/identify cache slots semantically (see module docstring);
        similarity ranking itself uses the exact vectorized dot product
        in `get()`, not this bucket.
        """
        dense = vec.toarray().reshape(-1) if sparse.issparse(vec) else np.asarray(vec).reshape(-1)
        self._ensure_planes(dense.shape[-1])
        bits = (self._simhash_planes @ dense) > 0
        return "".join("1" if b else "0" for b in bits)

    # -- writes -------------------------------------------------------------

    def put(self, canonical_query: str, query_variations: List[str], response: dict,
            device: Optional[str] = None, context_key: Optional[str] = None) -> None:
        texts = [canonical_query] + list(query_variations)
        vecs = self.embedder.encode_sparse(texts)  # already L2-normalized, sparse

        with self._lock:
            entry_idx = len(self._entries)
            device_key = _normalize_device(device)
            self._entries.append(CacheEntry(
                canonical_query=canonical_query,
                query_variations=list(query_variations),
                response=copy.deepcopy(response),
                device=device_key,
                context_key=context_key,
            ))
            self._pending.append(vecs)
            self._row_entry.extend([entry_idx] * vecs.shape[0])
            self._row_device.extend([device_key] * vecs.shape[0])
            self._row_context.extend([context_key] * vecs.shape[0])

            if len(self._entries) > self.max_entries:
                self._evict_oldest()

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        parts = ([self._matrix] if self._matrix is not None else []) + self._pending
        self._matrix = sparse.vstack(parts, format="csr")
        self._pending = []

    def _evict_oldest(self) -> None:
        # simple eviction of the least-recently-useful entry, keeping the
        # cache bounded for a long-running service
        victim_idx = min(range(len(self._entries)), key=lambda i: (self._entries[i].hits, self._entries[i].created_at))
        self._entries.pop(victim_idx)
        self._flush_pending()
        keep_rows = [r for r, e in enumerate(self._row_entry) if e != victim_idx]
        self._matrix = self._matrix[keep_rows] if self._matrix is not None else None
        self._row_entry = [e if e < victim_idx else e - 1 for e in self._row_entry if e != victim_idx]
        self._row_device = [self._row_device[r] for r in keep_rows]
        self._row_context = [self._row_context[r] for r in keep_rows]

    # -- reads ----------------------------------------------------------

    def get(self, query: str, query_variations: Optional[List[str]] = None,
            device: Optional[str] = None, context_key: Optional[str] = None) -> Optional[CacheResult]:
        """context_key: when the caller has its own reference text (a
        siis_response), pass its fingerprint -- only entries built from that
        same reference (or with no recorded fingerprint) can hit. Otherwise
        two different complaints that normalize to the same canonical text
        would be served an answer extracted from a DIFFERENT article than the
        one supplied in the request. None = no reference given, match any
        entry (the brief's "semantic lookup against pre-warmed entries").
        """
        t0 = time.perf_counter()
        with self._lock:
            if not self._entries:
                return None
            self._flush_pending()
            if self._matrix is None or self._matrix.shape[0] == 0:
                return None

            device_key = _normalize_device(device)
            query_texts = [query] + list(query_variations or [])
            query_vecs = self.embedder.encode_sparse(query_texts)  # (n_query, dim), normalized

            # One vectorized sparse-dense matmul across every stored vector at
            # once — this is the whole reason lookups stay fast past a few
            # thousand entries instead of a per-entry Python loop.
            sims = self._matrix @ query_vecs.T  # sparse result, (n_rows, n_query)
            best_per_row = np.asarray(sims.max(axis=1).todense()).reshape(-1)  # (n_rows,)

            # Hard device gate: "unknown" on either side matches anything,
            # otherwise the row is masked out of consideration entirely.
            compat = {d: _devices_compatible(d, device_key) for d in set(self._row_device)}
            device_mask = np.fromiter(
                (compat[d] for d in self._row_device), dtype=bool, count=len(self._row_device)
            )
            if context_key is not None:
                row_ctx = self._row_context
                ctx_mask = np.fromiter(
                    (c == context_key for c in row_ctx), dtype=bool, count=len(row_ctx)
                )
                device_mask = device_mask & ctx_mask
            best_per_row = np.where(device_mask, best_per_row, -1.0)

            # Group max similarity per entry (a query can match any of an
            # entry's canonical/variation rows).
            n_entries = len(self._entries)
            best_per_entry = np.full(n_entries, -1.0, dtype=np.float64)
            row_entry_arr = np.asarray(self._row_entry)
            np.maximum.at(best_per_entry, row_entry_arr, best_per_row)

            best_entry_idx = int(np.argmax(best_per_entry))
            best_sim = float(best_per_entry[best_entry_idx])

            latency_ms = (time.perf_counter() - t0) * 1000

            if best_sim < self.similarity_threshold:
                return None  # low-confidence -> null, never a guessed hit

            entry = self._entries[best_entry_idx]
            entry.hits += 1
            return CacheResult(
                response=copy.deepcopy(entry.response),  # callers may mutate; never alias the cached copy
                similarity=best_sim,
                matched_query=entry.canonical_query,
                latency_ms=latency_ms,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
