"""
Integration tests for pipeline.py — specifically the confidence gate on
the cache, which exists to prevent a concrete hallucination risk: two
different low-confidence (unclassifiable) queries for the same device
normalize to nearly identical canonical text, so caching either one's
answer would let it leak into the other.
"""
from cache import SemanticCache
from embeddings import TfidfEmbedder
from pipeline import Pipeline
from schema import Action, ContextDeeplinkResponse, Goal, StepGroup


def _fake_confident_stage1(siis_response: dict, enrichment) -> ContextDeeplinkResponse:
    return ContextDeeplinkResponse(contexts=[
        Goal(
            goal="Fix it",
            title=enrichment.symptom_label,
            score=0.9,
            actions=[Action(
                actionName="Do the thing",
                description="does the thing",
                stepGroups=[StepGroup(steps=["step one"])],
            )],
        )
    ])


def _identity_stage2(structured: ContextDeeplinkResponse, enrichment) -> ContextDeeplinkResponse:
    return structured


def _fresh_pipeline() -> Pipeline:
    cache = SemanticCache(TfidfEmbedder.load())
    return Pipeline(cache=cache, stage1_fn=_fake_confident_stage1, stage2_fn=_identity_stage2)


def _fresh_pipeline_with_default_stages() -> Pipeline:
    """A pipeline using the real not-wired-yet Stage 1/2 placeholders
    (pipeline.py's own _stage1_not_wired / _stage2_not_wired), i.e. what the
    API actually runs before Member B / Member C's code lands — genuinely
    empty contexts, not a test double standing in for "no viable solution".
    """
    cache = SemanticCache(TfidfEmbedder.load())
    return Pipeline(cache=cache)


def test_confident_query_is_cached_and_hit_on_repeat():
    pipeline = _fresh_pipeline()
    q = "My Galaxy S22 battery drains extremely fast, dead by noon even with light use."

    first = pipeline.run(q, {"title": "t", "content": "c"})
    assert first["meta"]["cache_hit"] is False
    assert first["enrichment"]["is_low_confidence"] is False
    assert first["enrichment"]["classification_source"] == "keyword_match"
    assert len(pipeline.cache) == 1

    second = pipeline.run("galaxy s22 battery dies so fast, gone by lunch", {"title": "t", "content": "c"})
    assert second["meta"]["cache_hit"] is True


def test_response_shape_matches_the_official_theme_brief_contract():
    """§5/Appendix B of the theme brief: query_variations is a TOP-LEVEL
    key (not nested inside enrichment), and cache_hit/latency_ms/model/
    cost_usd live under a "meta" object, not as flat top-level fields.
    Checking the exact envelope shape directly so a future refactor can't
    silently drift back to the old flat shape without a test noticing.
    """
    pipeline = _fresh_pipeline()
    result = pipeline.run(
        "My Galaxy S22 battery drains extremely fast, dead by noon even with light use.",
        {"title": "t", "content": "c"},
    )
    assert isinstance(result["query_variations"], list)
    from enrichment import MIN_VARIATIONS
    assert len(result["query_variations"]) >= MIN_VARIATIONS
    assert set(result["meta"].keys()) == {"cache_hit", "similarity", "latency_ms", "model", "cost_usd"}
    assert result["meta"]["model"] == "mock"  # conftest strips real LLM_PROVIDER for the whole suite
    assert result["meta"]["cost_usd"] == 0.0
    assert "cache_hit" not in result  # must NOT also be flat at the top level
    assert "latency_ms" not in result


def test_low_confidence_query_is_never_cached():
    pipeline = _fresh_pipeline()
    ambiguous = "asdkjfhaskdjfh qwerty"

    result = pipeline.run(ambiguous, {"title": "t", "content": "c"})
    assert result["meta"]["cache_hit"] is False
    assert result["enrichment"]["is_low_confidence"] is True
    assert len(pipeline.cache) == 0  # nothing written


def test_empty_result_carries_no_match_fallback_when_siis_response_given():
    """§4.2.3, non-negotiable: 'If the reference data contains no viable
    solution, the engine must return an empty list (contexts: []) with
    fallback metadata (\"fallback\": \"no_match\")'. Using the real
    not-wired-yet Stage 1 placeholder (returns empty contexts) with a
    genuine siis_response present, so this is "looked, found nothing".
    """
    pipeline = _fresh_pipeline_with_default_stages()
    result = pipeline.run(
        "My Galaxy S22 battery drains extremely fast, dead by noon even with light use.",
        {"title": "t", "content": "c"},
    )
    assert result["response"]["contexts"] == []
    assert result["response"]["fallback"] == "no_match"


def test_empty_result_carries_no_siis_context_fallback_when_nothing_given():
    """§8 Phase 4 names a second, distinct fallback reason: no siis_response
    at all means there was nothing to extract from in the first place —
    "no_siis_context", not "no_match" (which implies a real attempt was made
    against real reference text and came back empty).
    """
    pipeline = _fresh_pipeline_with_default_stages()
    result = pipeline.run(
        "My Galaxy S22 battery drains extremely fast, dead by noon even with light use.",
        {},
    )
    assert result["response"]["contexts"] == []
    assert result["response"]["fallback"] == "no_siis_context"


def test_fallback_metadata_is_absent_when_contexts_are_non_empty():
    """The fallback key must only appear on a genuinely empty result — never
    tacked onto a real, populated response, which would misrepresent a
    successful answer as a fallback case.
    """
    pipeline = _fresh_pipeline()
    result = pipeline.run(
        "My Galaxy S22 battery drains extremely fast, dead by noon even with light use.",
        {"title": "t", "content": "c"},
    )
    assert result["response"]["contexts"]  # non-empty (the fake stage1 always returns one Goal)
    assert "fallback" not in result["response"]


def test_llm_fallback_classified_query_still_never_enters_the_cache():
    """The guarded LLM fallback in enrichment.py (see
    test_llm_fallback_classifies_a_complaint_the_keyword_matcher_could_not)
    gives Stage 1/2 a real category instead of the generic unclassified text
    — but it's deliberately scored below the confidence threshold, so it
    must still be invisible to pipeline.py's cache gate. Proving that
    composition end-to-end rather than trusting it by inspection: a fake LLM
    resolves the category, Stage 1/2 run and get a real answer, but nothing
    is cached and a repeat of the exact same query recomputes fresh rather
    than serving a (potentially wrong, LLM-guessed) cached response.
    """
    import json
    from enrichment import enrich
    from llm_client import LLMClient

    class FakeLLM(LLMClient):
        def complete(self, system_prompt, user_prompt):
            if "category" in system_prompt:
                return json.dumps({"category": "keyboard_typing_problem"})
            return json.dumps([])  # variation-generation call: contribute nothing extra

    cache = SemanticCache(TfidfEmbedder.load())
    fake_llm = FakeLLM()
    pipeline = Pipeline(
        cache=cache, stage1_fn=_fake_confident_stage1, stage2_fn=_identity_stage2, llm_client=fake_llm,
    )
    # phrased to still miss the keyword taxonomy (no "keyboard" or "when I
    # type"-style phrase, which enrichment.py's taxonomy now catches directly)
    # so this still exercises the LLM fallback path being tested here.
    query = "My Galaxy S23 messages come out full of random symbols instead of the letters I actually pressed."

    # sanity check the fallback actually fires for this query before trusting the pipeline result
    enrichment = enrich(query, llm_client=fake_llm)
    assert enrichment.classification_source == "llm_fallback"
    assert enrichment.symptom_category == "keyboard_typing_problem"

    result = pipeline.run(query, {"title": "t", "content": "c"})
    assert result["enrichment"]["is_low_confidence"] is True
    assert result["response"]["contexts"][0]["title"] == enrichment.symptom_label  # Stage 1 got the real category
    assert result["meta"]["cache_hit"] is False
    assert len(pipeline.cache) == 0  # the whole point: llm_fallback never gets cached


def test_two_unrelated_low_confidence_queries_never_cross_contaminate():
    """The concrete hallucination risk this gate exists for: two genuinely
    different, both-unclassifiable complaints for the same (unknown)
    device normalize to nearly identical canonical text. Without the
    confidence gate, the second call would get served the first call's
    unrelated stage1/2 answer as a cache 'hit'.
    """
    call_count = {"n": 0}

    def _stage1_tagged(siis_response: dict, enrichment) -> ContextDeeplinkResponse:
        call_count["n"] += 1
        return ContextDeeplinkResponse(contexts=[
            Goal(goal=f"answer-{call_count['n']}", title="t", score=0.5, actions=[])
        ])

    cache = SemanticCache(TfidfEmbedder.load())
    pipeline = Pipeline(cache=cache, stage1_fn=_stage1_tagged, stage2_fn=_identity_stage2)

    r1 = pipeline.run("my phone does a weird thing sometimes idk", {"title": "t", "content": "c"})
    r2 = pipeline.run("something is off with it but not sure what", {"title": "t", "content": "c"})

    assert r1["response"]["contexts"][0]["goal"] == "answer-1"
    assert r2["response"]["contexts"][0]["goal"] == "answer-2"  # NOT answer-1 — no cross-contamination
    assert r2["meta"]["cache_hit"] is False
    assert len(cache) == 0


def test_mock_answers_are_never_cached(monkeypatch):
    """Offline-mock output (e.g. hosted Render asleep in hybrid mode) is a
    degraded echo of the article; caching it would serve it for every later
    paraphrase and block the real answer. Production default: not cached."""
    monkeypatch.delenv("CACHE_MOCK_ANSWERS", raising=False)
    pipeline = _fresh_pipeline()
    q = "My Galaxy S22 battery drains extremely fast, dead by noon even with light use."
    first = pipeline.run(q, {"title": "t", "content": "c"})
    assert first["response"]["contexts"]
    assert len(pipeline.cache) == 0
    second = pipeline.run(q, {"title": "t", "content": "c"})
    assert second["meta"]["cache_hit"] is False
