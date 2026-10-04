"""Regressions found by throwing unusual inputs at the running API."""
from enrichment import enrich, extract_symptom
from scrubber import _scrub_text, scrub_response


def test_link_only_sentences_leave_no_fragments():
    assert _scrub_text("Visit https://x.example/fix or www.samsung.com.")[0] == ""
    assert _scrub_text("Open Settings. See [guide](http://x.y).")[0] == "Open Settings."
    assert _scrub_text("Tap Display (more at samsung.com) and turn on Touch sensitivity.")[0] == \
        "Tap Display and turn on Touch sensitivity."
    assert _scrub_text("Read the [Display settings](http://a.b) page and enable Touch sensitivity.")[0] == \
        "Read the Display settings page and enable Touch sensitivity."
    # numbers that look like domains are not touched
    assert _scrub_text("Tap Battery > Charging (it is 3.5 times faster).")[0] == \
        "Tap Battery > Charging (it is 3.5 times faster)."


def test_no_markdown_or_url_survives_in_a_full_response():
    out = scrub_response({"contexts": [{
        "goal": "Follow these steps to perform this Touch Troubleshooting", "title": "Touch lag", "score": 0.9,
        "actions": [{"actionName": "Touch Settings", "description": "It will improve touch response time.",
                     "category": "auto",
                     "stepGroups": [{"steps": ["Visit https://evil.example.com/fix or www.samsung.com.",
                                               "Open Settings.", "See [guide](http://x.y)."]}]}]}]})
    text = str(out)
    for bad in ("http", "www.", "](", "samsung.com"):
        assert bad not in text
    assert out["contexts"][0]["actions"][0]["stepGroups"][0]["steps"] == ["Open Settings."]


def test_unclassified_complaint_gets_paraphrases_of_its_own_words():
    e = enrich("my phone is acting weird")
    assert e.symptom_category == "unclassified_issue"
    assert 8 <= len(e.query_variations) <= 10
    assert all("could not be automatically classified" not in v for v in e.query_variations)
    assert sum("acting weird" in v.lower() for v in e.query_variations) >= 5


def test_no_device_device_in_any_variation():
    for q in ["", "📱", "my phone is acting weird", "phone screen black wont turn on"]:
        assert all("device device" not in v and "device's device" not in v for v in enrich(q).query_variations)


def test_screen_lagging_is_touch_lag_but_black_screen_wins():
    assert extract_symptom("My TechCorp Nexa screen is lagging badly")[0].category == "touch_lag"
    assert extract_symptom("screen went black and lagging before")[0].category == "screen_blank_black"


def test_warm_file_rows_keep_their_own_siis_fingerprint(tmp_path):
    import json
    from cache import SemanticCache
    from embeddings import TfidfEmbedder
    from pipeline import Pipeline, siis_fingerprint
    from schema import ContextDeeplinkResponse
    siis = {"title": "Touch", "content": "Open Settings. Tap Display."}
    plan = {"contexts": [{"goal": "Follow these steps to perform this Touch Troubleshooting", "title": "Touch lag",
                          "score": 0.9, "actions": [{"actionName": "Touch Settings",
                          "description": "It will improve touch response time.", "category": "manual",
                          "stepGroups": [{"steps": ["Open Settings."]}]}]}]}
    f = tmp_path / "warm.jsonl"
    f.write_text(json.dumps({"query": "My Nexa A14 screen is lagging", "query_variations": [],
                             "response": plan, "siis_fingerprint": siis_fingerprint(siis)}) + "\n", encoding="utf-8")
    p = Pipeline(cache=SemanticCache(TfidfEmbedder.load()),
                 stage1_fn=lambda s, e: ContextDeeplinkResponse(contexts=[]), stage2_fn=lambda s, e: s)
    assert p.warm_from_results(f) == 1
    assert p.run("nexa a14 screen lags a lot", siis, allow_remote=False)["meta"]["cache_hit"] is True   # same article
    assert p.run("nexa a14 screen lags a lot", {}, allow_remote=False)["meta"]["cache_hit"] is True     # no article
    other = {"title": "x", "content": "Something else entirely."}
    assert p.run("nexa a14 screen lags a lot", other, allow_remote=False)["meta"]["cache_hit"] is False
