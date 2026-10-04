"""Hybrid mode: no local key -> cache misses go to the hosted deployment."""
import remote_client
from cache import SemanticCache
from embeddings import TfidfEmbedder
from pipeline import Pipeline
from schema import ContextDeeplinkResponse

REMOTE_ANSWER = {
    "response": {"contexts": [{
        "goal": "Follow these steps to perform this Battery Troubleshooting",
        "title": "Battery drain", "score": 0.9,
        "actions": [{"actionName": "Check Battery Usage", "description": "It will show battery usage details.",
                     "stepGroups": [{"steps": ["Open Settings.", "Tap Battery."]}], "category": "auto"}],
    }]},
    "meta": {"model": "gemini-3.5-flash-lite", "cost_usd": 0.0002},
}


def _pipe(calls):
    def local_stage1(siis, enrichment):
        calls.append("local")
        return ContextDeeplinkResponse(contexts=[])
    return Pipeline(cache=SemanticCache(TfidfEmbedder.load()), stage1_fn=local_stage1, stage2_fn=lambda s, e: s)


def test_no_key_forwards_miss_to_remote_and_caches_it(monkeypatch):
    monkeypatch.setenv("REMOTE_API_URL", "https://example.invalid")
    sent = []
    monkeypatch.setattr(remote_client, "call_remote", lambda q, s: sent.append(q) or REMOTE_ANSWER)
    calls = []
    p = _pipe(calls)
    q = "My Nexa X1 battery drains really fast"
    r = p.run(q, {"title": "t", "content": "Open Settings. Tap Battery."})
    assert r["meta"]["model"] == "remote:gemini-3.5-flash-lite"
    assert r["meta"]["cost_usd"] == 0.0002
    assert r["response"]["contexts"][0]["title"] == "Battery drain"
    assert calls == [] and sent == [q]
    again = p.run(q, {"title": "t", "content": "Open Settings. Tap Battery."})
    assert again["meta"]["cache_hit"] is True and sent == [q]  # repeat served locally


def test_explicit_mock_or_forwarded_request_never_forwards(monkeypatch):
    monkeypatch.setenv("REMOTE_API_URL", "https://example.invalid")
    monkeypatch.setattr(remote_client, "call_remote", lambda q, s: (_ for _ in ()).throw(AssertionError("forwarded")))
    calls = []
    _pipe(calls).run("My Nexa X1 battery drains really fast", {"content": "x"}, allow_remote=False)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    _pipe(calls).run("My Nexa X1 battery drains really fast", {"content": "x"})
    assert calls == ["local", "local"]


def test_remote_unreachable_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("REMOTE_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("REMOTE_TIMEOUT_SECONDS", "2")
    calls = []
    r = _pipe(calls).run("My Nexa X1 battery drains really fast", {"content": "x"})
    assert calls == ["local"] and r["meta"]["model"] == "mock"


def test_hosted_service_never_forwards(monkeypatch):
    monkeypatch.setenv("REMOTE_API_URL", "https://example.invalid")
    monkeypatch.setenv("RENDER", "true")
    assert remote_client.remote_url() is None
