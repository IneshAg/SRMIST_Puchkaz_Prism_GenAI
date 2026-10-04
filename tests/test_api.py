"""
Tests for api.py -- the actual FastAPI layer /v1/troubleshoot is served
through. Before this file, all 125+ tests in this suite exercised
enrichment.py/cache.py/pipeline.py directly and never went through a single
HTTP request, so a bug anywhere in request parsing, response serialization,
or an unhandled exception surfacing as a raw traceback would have shipped
completely untested on the team's actual deliverable endpoint.
"""
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client():
    return TestClient(api.app, raise_server_exceptions=False)


def test_health_and_healthz_ok(client):
    for path in ("/health", "/healthz"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_troubleshoot_happy_path_returns_full_contract_shape(client):
    r = client.post("/v1/troubleshoot", json={
        "query": "My Galaxy S22 battery drains extremely fast, dead by noon even with light use.",
        "siis_response": {"title": "Battery", "content": "Check battery usage in Settings."},
    })
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) >= {"query", "query_variations", "response", "meta", "enrichment"}
    assert isinstance(body["query_variations"], list) and len(body["query_variations"]) >= 1
    assert set(body["meta"].keys()) == {"cache_hit", "similarity", "latency_ms", "model", "cost_usd"}
    assert body["enrichment"]["symptom_category"] == "battery_drain"


def test_troubleshoot_missing_query_field_is_a_clean_422_not_a_500(client):
    r = client.post("/v1/troubleshoot", json={})
    assert r.status_code == 422


def test_troubleshoot_wrong_type_query_is_a_clean_422_not_a_500(client):
    r = client.post("/v1/troubleshoot", json={"query": 12345})
    assert r.status_code == 422


def test_troubleshoot_accepts_plain_string_siis_response_like_the_brief_example(client):
    # The brief's request example sends siis_response as a bare string.
    r = client.post("/v1/troubleshoot", json={"query": "battery drain", "siis_response": "Check Battery usage in Settings."})
    assert r.status_code == 200


def test_troubleshoot_malformed_siis_response_is_a_clean_422_not_a_500(client):
    r = client.post("/v1/troubleshoot", json={"query": "battery drain", "siis_response": 12345})
    assert r.status_code in (200, 422)  # pydantic may coerce; must never be a 500
    r = client.post("/v1/troubleshoot", json={"query": "battery drain", "siis_response": ["a", "b"]})
    assert r.status_code == 422


def test_troubleshoot_empty_query_does_not_crash():
    """An empty/whitespace-only query carries no information Stage 0 can
    classify, but that must surface as a normal low-confidence response
    (fallback: "no_siis_context"/"no_match"), never a 500 -- a judge
    trying an empty string is a completely plausible thing to try.
    """
    client = TestClient(api.app, raise_server_exceptions=False)
    for query in ("", "   "):
        r = client.post("/v1/troubleshoot", json={"query": query})
        assert r.status_code == 200
        assert r.json()["response"].get("fallback") == "no_siis_context"


def test_enrich_endpoint_returns_stage0_output_only(client):
    r = client.post("/v1/enrich", json={"query": "My Galaxy S24 Ultra won't charge at all."})
    assert r.status_code == 200
    body = r.json()
    assert body["symptom_category"] == "charging_fails_or_slow"
    # Stage 0 only -- no "response"/"meta" pipeline wrapper at this endpoint.
    assert "response" not in body and "meta" not in body


def test_cache_stats_reports_similarity_threshold_and_starts_empty():
    # a fresh Pipeline (not the module-level api.pipeline, which accumulates
    # entries across other tests in this file) so the starting count is known.
    from pipeline import Pipeline

    fresh_app_pipeline = Pipeline()
    original = api.pipeline
    api.pipeline = fresh_app_pipeline
    try:
        client = TestClient(api.app, raise_server_exceptions=False)
        r = client.get("/v1/cache/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["entries"] == 0
        assert body["similarity_threshold"] == fresh_app_pipeline.cache.similarity_threshold
    finally:
        api.pipeline = original


def test_unhandled_exception_returns_clean_500_not_a_raw_traceback(client, monkeypatch):
    """The actual regression test for the bug this file exists to catch:
    before api.py had a global exception handler, any exception raised
    inside pipeline.run() (a bug in enrichment/cache, or -- once wired in --
    Stage 1/2 blowing up on unexpected input) propagated straight out as
    FastAPI's default response: a raw Python stack trace, with file paths
    and source lines, served to whoever sent the request. Forcing exactly
    that here with a pipeline that always raises, and asserting the caller
    gets a clean, bounded JSON error instead.
    """
    def _boom(query, siis_response):
        raise RuntimeError("simulated Stage 1/2 failure")

    monkeypatch.setattr(api.pipeline, "run", _boom)
    r = client.post("/v1/troubleshoot", json={"query": "My Galaxy S22 battery drains fast."})
    assert r.status_code == 500
    body = r.json()
    assert body == {
        "error": "internal_error",
        "detail": "An unexpected error occurred while processing the request.",
    }
    # the failure mode this guards against: internals (file paths, source
    # lines, the exception's own message) leaking into the response body.
    assert "Traceback" not in r.text
    assert "simulated Stage 1/2 failure" not in r.text
    assert "enrichment.py" not in r.text and "pipeline.py" not in r.text


def test_troubleshoot_integration_sample_from_input_with_matching_siis_record(client):
    """Integration test: post a sample from data/input.txt with a matching record
    from data/siis_responses.json to POST /v1/troubleshoot, and assert non-empty
    contexts and full schema validity with Stage 2 stub deeplinks attached.
    """
    import json
    from pathlib import Path
    from schema import ContextDeeplinkResponse, actionCategory

    data_dir = Path(__file__).resolve().parent.parent / "data"
    with open(data_dir / "input.txt", "r", encoding="utf-8") as f:
        input_lines = [line.strip() for line in f if line.strip()]

    with open(data_dir / "siis_responses.json", "r", encoding="utf-8") as f:
        siis_data = json.load(f)

    # Sample from data/input.txt with matching record in data/siis_responses.json (row_21)
    matching_row = next(r for r in siis_data["responses"] if r["id"] == "row_21")
    sample_query = matching_row["original_query"]
    assert sample_query in input_lines

    r = client.post(
        "/v1/troubleshoot",
        json={"query": sample_query, "siis_response": matching_row["siis_response"]},
    )
    assert r.status_code == 200
    body = r.json()

    # Assert non-empty contexts
    contexts = body["response"]["contexts"]
    assert len(contexts) > 0

    # Assert schema validity via Pydantic model
    validated = ContextDeeplinkResponse.model_validate(body["response"])
    assert len(validated.contexts) == len(contexts)

    for context in validated.contexts:
        assert context.goal.startswith("Follow these steps to perform this ")
        assert 2 <= len(context.title.split()) <= 3
        assert 0.0 <= context.score <= 1.0
        assert len(context.actions) > 0
        for action in context.actions:
            assert action.actionName
            assert action.description.startswith("It will")
            assert 5 <= len(action.description.split()) <= 7
            assert action.category in (actionCategory.auto, actionCategory.manual, actionCategory.critical)
            assert len(action.stepGroups) > 0
            for group in action.stepGroups:
                assert len(group.steps) > 0
                if action.category == actionCategory.auto:
                    assert group.actionableDeeplink is not None
                    # verbatim from the catalog: a real match or its generic placeholder
                    catalog = {d["deeplink"] for d in json.load(open(data_dir / "deeplinks.json", encoding="utf-8"))["deeplinks"]}
                    assert group.actionableDeeplink.deeplink in catalog
                else:
                    assert group.actionableDeeplink is None


def test_troubleshoot_empty_contexts_fallback_no_match_when_siis_given(client, monkeypatch):
    """Verify that an empty-contexts result carries fallback 'no_match' when siis_response is given."""
    from schema import ContextDeeplinkResponse

    # Simulate Stage 1 finding no matching procedure from reference data
    monkeypatch.setattr(api.pipeline, "stage1_fn", lambda siis, enrich: ContextDeeplinkResponse(contexts=[]))

    r = client.post(
        "/v1/troubleshoot",
        json={
            "query": "My phone is acting strange",
            "siis_response": {"title": "General Info", "content": "Irrelevant reference content."},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["response"]["contexts"] == []
    assert body["response"]["fallback"] == "no_match"


def test_troubleshoot_empty_contexts_fallback_no_siis_context_when_no_siis(client, monkeypatch):
    """Verify that an empty-contexts result carries fallback 'no_siis_context' when no siis_response is provided."""
    # Cold cache: siis-less requests are now allowed to hit entries written by
    # earlier tests (that is the intended fast path), so isolate this one.
    import api
    from cache import SemanticCache
    from embeddings import TfidfEmbedder
    monkeypatch.setattr(api.pipeline, "cache", SemanticCache(TfidfEmbedder.load()))
    # 1. Without siis_response field
    r1 = client.post("/v1/troubleshoot", json={"query": "My phone battery drains very quickly"})
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["response"]["contexts"] == []
    assert body1["response"]["fallback"] == "no_siis_context"

    # 2. With siis_response set to None
    r2 = client.post(
        "/v1/troubleshoot",
        json={"query": "My phone battery drains very quickly", "siis_response": None},
    )
    assert r2.status_code == 200
    assert r2.json()["response"]["contexts"] == []
    assert r2.json()["response"]["fallback"] == "no_siis_context"

    # 3. With empty title and content
    r3 = client.post(
        "/v1/troubleshoot",
        json={"query": "My phone battery drains very quickly", "siis_response": {"title": "", "content": ""}},
    )
    assert r3.status_code == 200
    assert r3.json()["response"]["contexts"] == []
    assert r3.json()["response"]["fallback"] == "no_siis_context"


def test_global_exception_handler_returns_clean_json_with_no_traceback(client, monkeypatch):
    """Confirm the global exception handler catches exceptions and returns clean JSON with no traceback."""
    def _explode(*args, **kwargs):
        raise ValueError("simulated internal engine crash with sensitive file path secret/path.py")

    monkeypatch.setattr(api.pipeline, "run", _explode)
    r = client.post("/v1/troubleshoot", json={"query": "My screen won't turn on."})
    assert r.status_code == 500
    assert r.headers["content-type"] == "application/json"
    body = r.json()
    assert body == {
        "error": "internal_error",
        "detail": "An unexpected error occurred while processing the request.",
    }
    assert "Traceback" not in r.text
    assert "simulated internal engine crash" not in r.text
    assert "secret/path.py" not in r.text
    assert "Traceback (most recent call last)" not in r.text


def test_troubleshoot_request_timeout_guard_prevents_hang(client, monkeypatch):
    """Verify that a request taking longer than REQUEST_TIMEOUT_SECONDS triggers the timeout guard and never hangs."""
    import time

    def _slow_run(*args, **kwargs):
        time.sleep(0.3)
        return {"query": "slow", "response": {"contexts": []}}

    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setattr(api.pipeline, "run", _slow_run)

    start_time = time.perf_counter()
    r = client.post("/v1/troubleshoot", json={"query": "Slow hanging query"})
    elapsed = time.perf_counter() - start_time

    assert r.status_code == 504
    assert elapsed < 0.25  # Aborted promptly, did not hang
    body = r.json()
    assert body["error"] == "timeout"
    assert "timed out" in body["detail"]
    assert "Traceback" not in r.text


def test_troubleshoot_cache_hit_returns_under_300ms(client):
    """Confirm the cache-hit path returns in 300ms or less via TestClient."""
    import time
    from pipeline import Pipeline

    fresh_pipeline = Pipeline()
    original_pipeline = api.pipeline
    api.pipeline = fresh_pipeline
    try:
        query = "My Galaxy S22 battery drains extremely fast, dead by noon even with light use."
        payload = {
            "query": query,
            "siis_response": {"title": "Battery Drain", "content": "Check Battery usage in device Settings."},
        }

        # Request 1: Warm the cache (cold miss)
        r1 = client.post("/v1/troubleshoot", json=payload)
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["meta"]["cache_hit"] is False
        assert len(fresh_pipeline.cache) == 1

        # Request 2: Repeat query -> Cache hit
        t0 = time.perf_counter()
        r2 = client.post("/v1/troubleshoot", json=payload)
        round_trip_ms = (time.perf_counter() - t0) * 1000

        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["meta"]["cache_hit"] is True
        assert body2["meta"]["similarity"] is not None
        assert body2["meta"]["cost_usd"] == 0.0

        # Assert internal pipeline latency is <= 300ms
        assert body2["meta"]["latency_ms"] <= 300.0, f"Cache-hit latency was {body2['meta']['latency_ms']}ms > 300ms"
        # Assert full HTTP round-trip latency is <= 300ms
        assert round_trip_ms <= 300.0, f"Full HTTP round-trip was {round_trip_ms}ms > 300ms"
    finally:
        api.pipeline = original_pipeline

def test_oversized_query_is_rejected_with_422(client):
    r = client.post("/v1/troubleshoot", json={"query": "screen black " * 1000})
    assert r.status_code == 422
    r = client.post("/v1/enrich", json={"query": "x" * 5000})
    assert r.status_code == 422


def test_oversized_siis_text_is_rejected_with_422(client):
    q = "Galaxy S22 screen is black"
    r = client.post("/v1/troubleshoot", json={"query": q, "siis_response": "a" * 25000})
    assert r.status_code == 422
    r = client.post("/v1/troubleshoot", json={"query": q, "siis_response": {"title": "t", "content": "a" * 25000}})
    assert r.status_code == 422


def test_normal_sized_input_is_still_accepted(client):
    r = client.post(
        "/v1/troubleshoot",
        json={"query": "Galaxy S22 screen is black", "siis_response": {"title": "t", "content": "a" * 5000}},
    )
    assert r.status_code == 200


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    api._rate_limiter.reset()
    try:
        codes = [client.post("/v1/enrich", json={"query": "Galaxy S22 screen is black"}).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert codes[3:] == [429, 429]
        r = client.post("/v1/enrich", json={"query": "x"})
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
        assert client.get("/health").status_code == 200  # health is never limited
    finally:
        api._rate_limiter.reset()

