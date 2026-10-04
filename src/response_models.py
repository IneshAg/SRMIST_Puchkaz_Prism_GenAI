"""
Pydantic response models for api.py's routes, matching pipeline.py's
actual return shape (see its "Request/response contract" docstring) —
not just the brief's minimal Appendix B example.

Import and set as response_model= on each route in api.py. This fixes
the "additionalProp1: {}" placeholder /docs currently shows under
"Successful Response" for every route, since none of them declare a
return type FastAPI can introspect (they're all typed `-> dict`).
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel

from schema import ContextDeeplinkResponse


class TroubleshootContextResponse(ContextDeeplinkResponse):
    """ContextDeeplinkResponse (contexts: List[Goal]) plus the extra
    top-level key pipeline.run() adds to `response` before returning:
    - fallback: only present when contexts is empty (§4.2.3 / §8 Phase 4).
    It MUST stay declared here — if `response` were typed as the bare
    ContextDeeplinkResponse from schema.py instead, FastAPI's response_model
    filtering would silently strip this key out of the real JSON on
    the way out, including "fallback", which the brief calls non-negotiable.
    """
    fallback: Optional[Literal["no_match", "no_siis_context"]] = None


class Meta(BaseModel):
    cache_hit: bool
    similarity: Optional[float] = None
    latency_ms: float
    model: str
    cost_usd: float


class EnrichmentInfo(BaseModel):
    """The smaller enrichment summary embedded in /v1/troubleshoot's
    response (pipeline.run()'s `enrichment_info` dict) — a subset of
    EnrichResponse below, without raw_query/canonical_query/query_variations
    since those are already available at the top level of the envelope.
    """
    device: str
    symptom_category: str
    device_confidence: float
    symptom_confidence: float
    overall_confidence: float
    is_low_confidence: bool
    classification_source: str


class TroubleshootResponse(BaseModel):
    query: str
    query_variations: List[str]
    response: TroubleshootContextResponse
    meta: Meta
    enrichment: EnrichmentInfo


class EnrichResponse(BaseModel):
    """Full shape of EnrichmentResult.to_dict(), returned as-is by
    POST /v1/enrich (Stage 0 standalone).
    """
    raw_query: str
    canonical_query: str
    device: str
    symptom_category: str
    symptom_label: str
    device_confidence: float
    symptom_confidence: float
    overall_confidence: float
    is_low_confidence: bool
    classification_source: str
    query_variations: List[str]


class CacheStatsResponse(BaseModel):
    entries: int
    similarity_threshold: float


class HealthResponse(BaseModel):
    status: str
