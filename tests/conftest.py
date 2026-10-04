import os
import sys
from pathlib import Path

import pytest

# api.py pre-warms its cache from results.jsonl at import; tests need a cold cache.
os.environ.setdefault("CACHE_WARM_FILE", "")
# ...and must never forward to the hosted deployment (hybrid mode).
os.environ["REMOTE_API_URL"] = ""

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _isolate_from_real_llm_config(monkeypatch):
    """The test suite must stay fast and deterministic regardless of what a
    developer's local .env happens to have configured. Without this, a real
    LLM_PROVIDER=gemini/openai in .env leaks into every test that
    doesn't explicitly inject a fake llm_client (most of them call
    get_llm_client() internally by default) — each one then attempts a real
    network call, and with a provider blocked on this network that's an
    8-second stall per test (bounded by _TimeoutGuardedClient, but still 8s
    x dozens of tests), not the ~4-second suite this is supposed to be.
    Found by exactly that happening: a full run that normally takes ~4s
    hung well past two minutes once a real .env existed on this machine.
    """
    monkeypatch.setenv("CACHE_MOCK_ANSWERS", "1")  # cache-mechanics tests run on the mock; prod never caches it
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import llm_client
    llm_client._client_cache.clear()  # don't let an earlier test's cached client leak in
