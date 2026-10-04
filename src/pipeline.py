"""
Glue for the full POST /v1/troubleshoot flow. This module owns Stage 0
(enrichment.py) and Stage 3 (cache.py) — Member A's scope — and defines
the two extension points Member B (Stage 1: LLM structuring) and
Member C (Stage 2: deeplink mapping) plug into.

Request/response contract
--------------------------
The request shape is confirmed directly against data/siis_responses.json's
own readme ("siis_response is the payload your API must accept in
POST /v1/troubleshoot") and its actual per-record shape
({"title": ..., "content": ...}) — the theme brief's own request example
shows a bare "<optional raw text context>" string, but the real provided
data is an object, and that's what this accepts.

The response shape matches the official theme brief's §5 API contract and
Appendix B worked example exactly (not just data/sample_output.json's
smaller shape, which predates the full brief): "query_variations" is a
TOP-LEVEL key (Stage 0's paraphrases), not buried inside "enrichment", and
"cache_hit"/"latency_ms"/"model"/"cost_usd" are nested under a "meta"
object, not flat top-level fields. "enrichment" (device/symptom/confidence
signals) isn't part of the official contract at all — it's kept as an
additive debug field since nothing in the spec forbids extra top-level
keys, and it's genuinely useful for demoing Stage 0 in isolation.

    POST /v1/troubleshoot
    {
      "query": "<raw customer complaint>",
      "siis_response": {"title": "...", "content": "..."}
    }

    -> {
      "query": "<raw customer complaint>",
      "query_variations": [ ... 8-10 paraphrases from Stage 0 ... ],
      "response": {"contexts": [ ... Goal objects, per schema.py ... ]},
      "meta": {"cache_hit": bool, "similarity": float | None,
                "latency_ms": float, "model": str, "cost_usd": float},
      "enrichment": {"device", "symptom_category", "device_confidence",
                     "symptom_confidence", "overall_confidence", "is_low_confidence",
                     "classification_source"}
    }

§4.2.3 of the brief is explicit and non-negotiable: "If the reference data
contains no viable solution, the engine must return an empty list
(contexts: []) with fallback metadata (\"fallback\": \"no_match\")" — and
the roadmap (§8, Phase 4) names a second reason, "no_siis_context", for
when there was no siis_response to work from at all. Both are added to
`response` (alongside "contexts") whenever Stage 1/2 come back empty,
distinguishing "nothing to look at" from "looked, found nothing".
"meta.cost_usd" reports the real per-request USD cost by reading each
provider's token-usage fields (Gemini: usage_metadata; OpenAI/:
usage) and applying published per-token rates. On the free tier, this is
mathematically correct but not billed — the figure represents the true
paid-tier equivalent cost and activates automatically when a paid key is
configured. MockLLMClient returns 0.0. Stage 0 + 1 + 2 costs are
accumulated via LLMClient.consume_cost() and summed per-request.

Flow
----
1. Stage 0 (enrichment.enrich): normalize the raw query, extract device,
   generate 8-10 query variations, and score how confident that
   normalization actually is.
2. Stage 3 (cache.SemanticCache.get): if a semantically-equivalent query
   for the same device was already answered, return that validated
   response immediately (this is the ≤300ms path) — but ONLY when Stage 0
   was confident about what it normalized. See "Why low confidence bypasses
   the cache entirely" below.
3. On a miss: Stage 1 (structure extraction from siis_response) then
   Stage 2 (deeplink mapping) run to build the response — these are
   Member B / Member C's stages, injected as callables so this module
   has zero hard dependency on their implementation landing first.
4. The freshly computed response is stored in the cache (keyed by the
   Stage 0 canonical query + variations + device) before being returned,
   so the next semantically-equivalent query is a cache hit — again, only
   when Stage 0 was confident.

Why low confidence bypasses the cache entirely
------------------------------------------------
When Stage 0 can't classify a symptom, every such query normalizes to
nearly the same canonical text regardless of what the customer actually
said (device + "is experiencing an issue that could not be automatically
classified..."). If that got cached, a completely unrelated future
low-confidence query — same device, genuinely different problem — could
register a false-positive cache HIT and be served someone else's answer.
That's a hallucination introduced by the cache layer, not by Stage 1/2, so
it's closed here: `enrichment.is_low_confidence` skips both the cache read
and the cache write, and Stage 1/2 run fresh every time until Stage 0 (or
a human) actually knows what's being asked. This does mean a low-confidence
query never gets the ≤300ms fast path — that's the correct trade: the
300ms budget is a promise about confidently-recognized repeat queries, not
a license to guess quickly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Callable, Optional

from cache import SemanticCache
from embeddings import TfidfEmbedder
from enrichment import EnrichmentResult, enrich
from llm_client import get_llm_client
from schema import ContextDeeplinkResponse
from scrubber import scrub_response
from structure_extraction import structure_extraction
from deeplink_mapping import deeplink_mapping
import remote_client

# -- Member B / Member C extension points --------------------------------
# Signature each stage must implement. `enrichment` is passed through so
# later stages can use the canonical query / device / variations without
# re-deriving them.
Stage1Fn = Callable[..., ContextDeeplinkResponse]  # (siis_response, enrichment, *, llm_client=None)
Stage2Fn = Callable[[ContextDeeplinkResponse, EnrichmentResult], ContextDeeplinkResponse]


def _stage1_not_wired(siis_response: dict, enrichment: EnrichmentResult) -> ContextDeeplinkResponse:
    """Placeholder Stage 1. Returns an explicitly-empty result rather than
    inventing troubleshooting steps, so a pipeline run before Member B's
    code lands is obviously incomplete instead of silently wrong.
    Replace via Pipeline(stage1_fn=<member B's function>).
    """
    return ContextDeeplinkResponse(contexts=[])


def _stage2_not_wired(structured: ContextDeeplinkResponse, enrichment: EnrichmentResult) -> ContextDeeplinkResponse:
    """Placeholder Stage 2 (deeplink mapping) — passthrough, no deeplinks
    attached. Replace via Pipeline(stage2_fn=<member C's function>).
    """
    return structured



def _has_siis(siis_response) -> bool:
    return bool(
        siis_response
        and (
            not isinstance(siis_response, dict)
            or any(str(v).strip() for v in siis_response.values())
        )
    )


def siis_fingerprint(siis_response) -> Optional[str]:
    """Stable short hash of the reference text, used so a cache hit is only
    served from an entry built on the SAME siis_response the caller sent."""
    if not _has_siis(siis_response):
        return None
    if isinstance(siis_response, dict):
        payload = json.dumps(
            {k: str(v).strip() for k, v in siis_response.items()}, sort_keys=True, ensure_ascii=False
        )
    else:
        payload = str(siis_response).strip()
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _norm_q(q: str) -> str:
    import re as _re
    q = _re.sub(r"^\s*\d+[.)]\s*", "", q or "").strip().strip('"\u201c\u201d\' ')
    return _re.sub(r"\s+", " ", q).lower()


def cache_eligible(enrichment: EnrichmentResult) -> bool:
    """Cache gate. Keyed on the SYMPTOM being keyword-recognized, not on
    min(device, symptom): a complaint that names no model ("phone screen went
    black", the brief's own keyword register) has a perfectly well-defined
    symptom and must still reach the fast path. The device gate is handled by
    the cache itself (unknown device = wildcard). Unclassified and
    llm_fallback results stay out, exactly as before."""
    return enrichment.classification_source == "keyword_match" and enrichment.symptom_confidence >= 0.5


def cache_device(enrichment: EnrichmentResult) -> Optional[str]:
    # "Samsung device" is enrichment's placeholder, NOT a device: pass None so
    # the cache treats it as its "unknown" wildcard instead of a literal
    # device string that only ever matches other placeholder entries.
    return enrichment.device if enrichment.device_confidence > 0 else None


class Pipeline:
    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        stage1_fn: Stage1Fn = structure_extraction,
        stage2_fn: Stage2Fn = deeplink_mapping,
        llm_client=None,
    ):
        self.cache = cache or SemanticCache(TfidfEmbedder.load())
        self.stage1_fn = stage1_fn
        self.stage2_fn = stage2_fn
        # Optional explicit injection (tests, or an app wiring a specific client).
        # If not given, enrich() falls back to get_llm_client(), which reads
        # LLM_PROVIDER from the environment — so the common case (just set the
        # env var, no code changes) keeps working unchanged.
        self.llm_client = llm_client

    def run(self, query: str, siis_response: dict, allow_remote: bool = True) -> dict:
        t0 = time.perf_counter()

        # Resolved once (not inside enrich()) so meta.model can report the
        # same client actually used for this request without a second,
        # redundant resolution — get_llm_client() is cached by (provider,
        # model) so this costs nothing extra either way, but resolving once
        # here keeps "which client answered" and "what enrich() used" the
        # same object by construction, not by coincidence.
        client = self.llm_client or get_llm_client()

        # Stage 0
        enrichment = enrich(query, llm_client=client)
        # Drain Stage 0 token cost (0.0 on free tier / mock; real USD on paid tier).
        # Must be called before Stage 1/2 so their tokens aren't double-counted.
        cost_usd = client.consume_cost()
        enrichment_info = {
            "device": enrichment.device,
            "symptom_category": enrichment.symptom_category,
            "device_confidence": enrichment.device_confidence,
            "symptom_confidence": enrichment.symptom_confidence,
            "overall_confidence": enrichment.overall_confidence,
            "is_low_confidence": enrichment.is_low_confidence,
            "classification_source": enrichment.classification_source,
        }

        # Stage 3 (read path) — skipped entirely when Stage 0 isn't confident;
        # see module docstring for why (false-positive hits across unrelated
        # low-confidence queries).
        has_siis = _has_siis(siis_response)
        context_key = siis_fingerprint(siis_response)
        eligible = cache_eligible(enrichment)
        # The raw complaint is embedded too: a paraphrase is often closer to
        # the customer's own wording than to the templated canonical text.
        cache_texts = [query] + list(enrichment.query_variations)
        cache_hit = None
        if eligible:
            cache_hit = self.cache.get(
                enrichment.canonical_query, cache_texts,
                device=cache_device(enrichment), context_key=context_key,
            )
        if cache_hit is not None:
            return {
                "query": query,
                "query_variations": enrichment.query_variations,
                "response": cache_hit.response,
                "meta": {
                    "cache_hit": True,
                    "similarity": cache_hit.similarity,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "model": client.model_name,
                    "cost_usd": cost_usd,  # Stage 0 only; cache hit = no Stage 1/2 cost
                },
                "enrichment": enrichment_info,
            }

        # Hybrid mode (remote_client.py): no local key -> forward the miss to
        # the hosted deployment that holds the Gemini key. Falls through to
        # the local stages if the remote is disabled or unreachable.
        model_name = client.model_name
        remote = None
        if allow_remote and remote_client.should_forward(client):
            remote = remote_client.call_remote(query, siis_response)
            try:
                if remote is not None:
                    ContextDeeplinkResponse.model_validate({"contexts": remote["response"].get("contexts", [])})
            except Exception:
                remote = None  # malformed remote answer: compute locally instead
        if remote is not None:
            response_dict = scrub_response(dict(remote["response"]))
            rmeta = remote.get("meta") or {}
            cost_usd += float(rmeta.get("cost_usd") or 0.0)
            model_name = f"remote:{rmeta.get('model', 'unknown')}"
        else:
            # Stage 1 + Stage 2 (Member B / Member C).
            # Pass llm_client to Stage 1 if it accepts it (so its tokens flow into
            # consume_cost() alongside Stage 0's). Falls back gracefully for test
            # stubs that don't declare the kwarg.
            import inspect as _inspect
            _s1_params = _inspect.signature(self.stage1_fn).parameters
            _s1_kwargs = {"llm_client": client} if "llm_client" in _s1_params or any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in _s1_params.values()
            ) else {}
            structured = self.stage1_fn(siis_response, enrichment, **_s1_kwargs)


            final = self.stage2_fn(structured, enrichment)
            # Drain any tokens Stage 1/2 spent (Member B/C call client.complete() here).
            # On free tier: 0.0. On paid: adds their costs to Stage 0's cost above.
            cost_usd += client.consume_cost()
            response_dict = final.model_dump()
            response_dict = scrub_response(response_dict)

            if self.stage2_fn is _stage2_not_wired:
                response_dict["deeplinks_pending"] = True

        # §4.2.3 (non-negotiable): an empty result must carry fallback
        # metadata, not just a bare empty list — "no_siis_context" when
        # there was nothing to extract from at all, "no_match" when Stage
        # 1/2 ran against real reference text but found no viable solution
        # (§8 Phase 4 names both reasons explicitly). Baked into
        # response_dict before caching so a later cache HIT on this same
        # (rare — see the confidence gate below) entry still carries it.
        if not response_dict.get("contexts"):
            response_dict["fallback"] = "no_match" if has_siis else "no_siis_context"

        # Stage 3 (write path) — same confidence gate as the read path,
        # otherwise this is exactly what would poison the cache with a
        # near-duplicate key for unrelated future ambiguous queries.
        # Empty/fallback results are never cached: a query first seen WITHOUT
        # a siis_response ("no_siis_context") would otherwise poison the entry
        # and every later request for it -- even one that does carry the
        # reference text -- would be served the empty answer from cache.
        if eligible and response_dict.get("contexts"):
            self.cache.put(
                enrichment.canonical_query, cache_texts, copy.deepcopy(response_dict),
                device=cache_device(enrichment), context_key=context_key,
            )

        return {
            "query": query,
            "query_variations": enrichment.query_variations,
            "response": response_dict,
            "meta": {
                "cache_hit": False,
                "similarity": None,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "model": model_name,
                "cost_usd": cost_usd,  # Stage 0 + Stage 1 + Stage 2 total
            },
            "enrichment": enrichment_info,
        }

    def warm_from_results(self, path, siis_file=None) -> int:
        """Pre-warm the semantic cache from an already-generated results.jsonl
        (validated responses, no LLM calls). The brief: "If siis_response is
        omitted, the engine performs semantic lookup against pre-warmed cache
        entries" -- without this a freshly started server has an empty cache
        and answers every siis-less request with no_siis_context.
        Stage 0 is re-run with the offline mock client (keyword path is
        deterministic, so the canonical/variations match what a live request
        computes). When the row's complaint is one of data/siis_responses.json's
        original queries, the entry is fingerprinted with that reference text,
        so a request that sends the same siis_response can also hit; siis-less
        requests match any entry. Returns the number of entries written.
        """
        from pathlib import Path
        from llm_client import MockLLMClient

        path = Path(path)
        if not path.is_file():
            return 0
        mock = MockLLMClient()
        siis_by_query = {}
        siis_path = Path(siis_file) if siis_file else path.parent / "data" / "siis_responses.json"
        if siis_path.is_file():
            try:
                for rec in json.loads(siis_path.read_text(encoding="utf-8")).get("responses", []):
                    key = _norm_q(rec.get("original_query", ""))
                    if key:
                        siis_by_query[key] = rec.get("siis_response") or {}
            except Exception:
                siis_by_query = {}
        written = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                response = row.get("response") or {}
                if not response.get("contexts"):
                    continue
                ContextDeeplinkResponse.model_validate({"contexts": response["contexts"]})
                query = row["query"]
                enrichment = enrich(query, llm_client=mock)
                if not cache_eligible(enrichment):
                    continue
                texts = [query] + list(enrichment.query_variations) + [
                    v for v in row.get("query_variations", []) if isinstance(v, str)
                ]
                self.cache.put(
                    enrichment.canonical_query, texts, response, device=cache_device(enrichment),
                    context_key=siis_fingerprint(siis_by_query.get(_norm_q(query))),
                )
                written += 1
            except Exception:  # a bad line must never stop the server from starting
                continue
        return written
