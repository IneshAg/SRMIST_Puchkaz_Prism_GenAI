"""
Pluggable LLM client for Stage 0 (query enrichment).

No provider/API key had been chosen yet when this was built, so this
module ships:
  * a common `LLMClient` interface
  * ready adapters for Gemini, OpenAI and  (each ~10 lines,
    activate by setting env vars — see below)
  * a dependency-free `MockLLMClient` fallback that uses templated
    paraphrasing rules instead of an API call, so the pipeline always
    returns a valid response even with no key configured (never crashes,
    never blocks the rest of the team on an API key)

Configure via environment variables:
    LLM_PROVIDER = "gemini" | "openai" | "" | "mock"   (default: mock)
    GOOGLE_API_KEY / OPENAI_API_KEY as needed
    LLM_MODEL   (optional override of the default model per provider)

Easiest way to set these: copy .env.example (project root) to .env and fill
in the two lines. python-dotenv (if installed — it's in requirements.txt)
loads .env automatically on import, below. No .env / no dotenv installed?
Nothing breaks — get_llm_client() falls back to the mock client exactly as
it always did. .env is gitignored so a real key never gets committed.

Whichever provider is used, `enrichment.py` always programmatically
validates the LLM's output (Stage 0's own "never trust the LLM to
self-constrain" rule, same principle Member B applies to Stage 1) — the
LLM proposes, the code disposes.
"""
from __future__ import annotations

import concurrent.futures
import os
import re
from abc import ABC, abstractmethod
from typing import List

try:
    from dotenv import load_dotenv  # optional dependency, see requirements.txt

    load_dotenv()  # no-op if there's no .env file; never raises
except ImportError:
    pass  # python-dotenv not installed — env vars still work if set another way


class LLMClient(ABC):
    """Minimal interface: give it a task, get raw text back."""

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._cost_usd = 0.0
        import threading
        instance._cost_lock = threading.Lock()
        return instance

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...

    @property
    def model_name(self) -> str:
        """Identifier for what actually produced a response, surfaced in the
        API's meta.model field (spec Appendix B's worked example shows
        "model": "gpt-4o-mini" there). Defaults to "mock" so a client that
        doesn't override this (MockLLMClient) still reports an honest,
        specific value instead of nothing.
        """
        return "mock"

    def consume_cost(self) -> float:
        """Return the USD cost accumulated across all complete() calls since
        the last consume_cost() call, then reset the counter to 0.0.

        Pipeline pattern — call once after each stage to drain costs into a
        running request total without double-counting:

            cost  = client.consume_cost()   # Stage 0 tokens
            stage1_fn(...)                  # Member B calls client.complete()
            cost += client.consume_cost()   # Stage 1 tokens
            stage2_fn(...)                  # Member C calls client.complete()
            cost += client.consume_cost()   # Stage 2 tokens
            # cost = true USD total for the full request

        Returns 0.0 on MockLLMClient (no API key, no cost) and on free-tier
        keys (providers return real token counts; the math gives the
        theoretical paid-tier cost, which auto-activates when you switch
        from free to paid without any code change).
        """
        with self._cost_lock:
            total = self._cost_usd
            self._cost_usd = 0.0
            return total

    def _record_usage(self, prompt_tokens: int, completion_tokens: int,
                      rate_in: float, rate_out: float) -> None:
        """Accumulate cost from a single API response's token-usage fields.

        Subclasses call this inside complete() after reading the provider's
        usage object. rate_in / rate_out are per-token USD rates for
        input / output respectively (see each provider subclass for the
        values used).

        Accumulated — not replaced — so multiple complete() calls within
        one pipeline request sum correctly before consume_cost() drains them.
        Never raises; cost stays at 0.0 if the provider doesn't return usage.
        """
        with self._cost_lock:
            self._cost_usd += (prompt_tokens * rate_in + completion_tokens * rate_out)



class MockLLMClient(LLMClient):
    """Deterministic, offline, rule-based stand-in for a real LLM.

    Used automatically when LLM_PROVIDER is unset/"mock", or as the
    automatic fallback if a configured provider errors out. It doesn't
    "understand" the complaint the way an LLM would, so `enrichment.py`
    layers real normalization/variation-generation rules on top rather
    than trusting this for anything beyond a text echo — this class only
    exists so the rest of the system never has a hard dependency on an
    API key being present.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "Structure Extraction" in system_prompt or "Goal" in system_prompt:
            import json
            import re

            title_m = re.search(r"Title:\s*\n?([^\n]+)", user_prompt)
            content_m = re.search(r"Content:\s*\n?([\s\S]+?)(?=\nThe reference|\nExtract|\nReturn|\Z)", user_prompt)
            title_text = title_m.group(1).strip() if title_m else "Device"
            content_text = content_m.group(1).strip() if content_m else "Check Settings."

            words = re.findall(r"\b[\w'-]+\b", title_text)
            if 2 <= len(words) <= 3:
                topic = " ".join(words[:2])
                title = " ".join(words)
            elif len(words) >= 4:
                topic = " ".join(words[:2])
                title = " ".join(words[:2])
            else:
                topic = words[0] if words else "Device"
                title = f"{topic} Troubleshooting"

            steps = []
            for line in content_text.splitlines():
                clean = line.strip().lstrip("#").strip()
                if clean and len(clean) > 5 and not clean.startswith("http"):
                    steps.append(clean)
                    if len(steps) >= 2:
                        break
            if not steps:
                steps = [content_text.splitlines()[0].strip() if content_text.splitlines() else "Open Settings."]

            mock_response = {
                "contexts": [
                    {
                        "goal": f"Follow these steps to perform this {topic} Troubleshooting",
                        "title": title,
                        "score": 0.9,
                        "actions": [
                            {
                                "actionName": f"{topic} Settings",
                                "description": "It will check device settings.",
                                "stepGroups": [
                                    {
                                        "steps": steps,
                                        "actionableDeeplink": None,
                                        "validationDeeplink": None,
                                    }
                                ],
                                "category": "auto",
                            }
                        ],
                    }
                ]
            }
            return json.dumps(mock_response)

        if "screen lag" in user_prompt.lower():
            return '{"category": "touch_lag", "variations": []}'
        elif "display" in user_prompt.lower() or "shattered" in user_prompt.lower():
            return '{"category": "display_issue", "variations": []}'
        elif "back up" in user_prompt.lower() or "backup" in user_prompt.lower():
            return '{"category": "cloud_sync", "variations": []}'
        return '{"category": "unclassified_issue", "variations": []}' 


# Every network call below gets an explicit, short timeout. Found by testing,
# not by inspection: a blocked/unreachable host (e.g. a provider whose domain
# isn't on this network's egress allowlist — the same class of issue that
# blocked huggingface.co for the embedding model) doesn't fail fast on its
# own here. A bare `curl` to it returns instantly with a clear 403 from the
# proxy, but the Gemini SDK's own retry/backoff logic swallowed that and hung
# well past 60 seconds with no exception raised. Since this call sits in
# every request's critical path (see pipeline.py's module docstring), an
# unbounded hang is far worse than a clean, fast failure into the mock
# fallback — it would silently blow the "fast" requirement instead of
# visibly degrading quality. 30 seconds is the limit — generous enough for
# Stage 1's heavier structure-extraction prompt (which is longer than
# Stage 0's short classification call) but short enough that a hung provider
# can't matter for the request latency budget.
_LLM_TIMEOUT_SECONDS = 30

# §6 grades "Deterministic Execution: consistent action plans for identical or
# semantically identical inputs". Every real provider is therefore called at
# temperature 0 (greedy decoding) — provider defaults (typically 1.0)
# sample, so two identical llm_fallback requests could come back
# with different categories/paraphrases run to run. Temperature 0 alone is
# necessary but not sufficient: providers don't guarantee bit-identical
# output even at T=0 (batching / floating-point nondeterminism on their
# side), so _TimeoutGuardedClient below also memoizes successful responses
# per exact (system_prompt, user_prompt) pair — an identical request within
# a process gets an identical answer by construction, not by hope.
_LLM_TEMPERATURE = 0.0
_LLM_MEMO_MAX_ENTRIES = 2048


class GeminiLLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        import google.generativeai as genai  # lazy import

        api_key = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        self._model_name = model or os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
        self._model = genai.GenerativeModel(self._model_name)

    # Gemini flash-lite pricing (per token, USD, as of 2025)
    # Source: https://ai.google.dev/gemini-api/docs/pricing
    # Free tier: same token counts returned, cost math gives $0.00 at 0 QPM cost.
    # Switch to paid: rates activate automatically with no code change needed.
    _RATE_IN  = 0.10 / 1_000_000   # $0.10 per 1M input tokens
    _RATE_OUT = 0.40 / 1_000_000   # $0.40 per 1M output tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._model.generate_content(
            [system_prompt, user_prompt],
            generation_config={"temperature": _LLM_TEMPERATURE},
            request_options={"timeout": _LLM_TIMEOUT_SECONDS},
        )
        try:
            u = resp.usage_metadata
            self._record_usage(
                u.prompt_token_count or 0,
                u.candidates_token_count or 0,
                self._RATE_IN, self._RATE_OUT,
            )
        except Exception:
            pass  # metadata absent on some response types — cost stays 0.0
        return resp.text

    @property
    def model_name(self) -> str:
        return self._model_name



class OpenAILLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=_LLM_TIMEOUT_SECONDS)
        self._model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")

    # gpt-4o-mini pricing (per token, USD, as of 2025)
    # Free tier / no billing: token counts still returned; rates activate on paid tier automatically.
    _RATE_IN  = 0.15 / 1_000_000   # $0.15 per 1M input tokens
    _RATE_OUT = 0.60 / 1_000_000   # $0.60 per 1M output tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=_LLM_TEMPERATURE,
            seed=0,  # OpenAI's best-effort reproducibility knob; harmless where ignored
        )
        try:
            self._record_usage(
                resp.usage.prompt_tokens or 0,
                resp.usage.completion_tokens or 0,
                self._RATE_IN, self._RATE_OUT,
            )
        except Exception:
            pass
        return resp.choices[0].message.content or ""

    @property
    def model_name(self) -> str:
        return self._model


class _TimeoutGuardedClient(LLMClient):
    """Wraps a real provider client and enforces a hard wall-clock timeout
    around complete() from the OUTSIDE, independent of whatever timeout
    parameter the underlying SDK claims to honor. Necessary because one
    isn't enough: found by testing that a blocked/unreachable host can make
    a provider's own internal retry/handshake logic hang far past its
    documented per-request timeout (observed directly: Gemini's gRPC
    transport hung 60+ seconds against a network that blocks its endpoint,
    despite `request_options={"timeout": 8}` on the call itself — the hang
    happens at a connection layer that parameter doesn't reach). Since this
    call sits in every request's critical path, an unbounded hang here is a
    much worse failure than a fast, clean fallback to the mock generator.

    A background thread that exceeds the timeout is abandoned, not killed
    (Python cannot forcibly stop a running thread) — but the caller is
    never blocked longer than `timeout_seconds` regardless, which is the
    property that actually matters for the API's own latency budget.

    Also memoizes successful responses per exact (system_prompt,
    user_prompt) pair (bounded, FIFO-evicted) — see _LLM_TEMPERATURE for
    why temperature 0 alone doesn't guarantee identical output. Failures
    and timeouts are never memoized (they raise before the store), so a
    transient provider error can't get pinned as "the" answer.
    """

    def __init__(self, inner: LLMClient, timeout_seconds: float = _LLM_TIMEOUT_SECONDS,
                 memo_max_entries: int = _LLM_MEMO_MAX_ENTRIES):
        self._inner = inner
        self._timeout = timeout_seconds
        self._memo: dict = {}
        self._memo_max = memo_max_entries

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        key = (system_prompt, user_prompt)
        if key in self._memo:
            return self._memo[key]

        # Use a fresh executor per call, not a shared one with max_workers=1.
        # If max_workers=1 is shared, a single timed-out call abandons its task
        # on that one worker, permanently starving all subsequent calls.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._inner.complete, system_prompt, user_prompt)
            text = future.result(timeout=self._timeout)  # raises TimeoutError past the deadline
            if len(self._memo) >= self._memo_max:
                self._memo.pop(next(iter(self._memo)))  # dicts keep insertion order -> FIFO
            self._memo[key] = text
            return text
        finally:
            # wait=False prevents blocking here if the task is still hanging
            executor.shutdown(wait=False, cancel_futures=True)


    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def consume_cost(self) -> float:
        """The inner provider client is the one whose complete() calls
        _record_usage(), so its counter is where the real token cost lands.
        Without this override the wrapper drained its OWN (always-zero)
        counter and meta.cost_usd was 0.0 for every real Gemini/OpenAI
         request.
        """
        return super().consume_cost() + self._inner.consume_cost()


# get_llm_client() is called on every enrich() invocation (see enrichment.py),
# so a real provider client — and the thread pool _TimeoutGuardedClient opens
# for it — is built once and reused, not reconstructed per request. Keyed by
# (provider, model) so a test that changes LLM_PROVIDER/LLM_MODEL mid-run
# still gets a fresh client rather than a stale cached one.
_client_cache: dict = {}


def get_llm_client() -> LLMClient:
    """Factory: picks the provider from LLM_PROVIDER, falls back to the
    offline mock if unset, unrecognized, or the provider fails to
    initialize (e.g. missing API key) — so a missing key degrades the
    quality of Stage 0's output but never breaks the API. Any real provider
    is wrapped in _TimeoutGuardedClient so a blocked/slow network degrades
    to the mock generator within `_LLM_TIMEOUT_SECONDS`, not an unbounded
    hang.
    """
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    cache_key = (provider, os.environ.get("LLM_MODEL"))
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    client: LLMClient
    try:
        if provider == "gemini":
            client = _TimeoutGuardedClient(GeminiLLMClient())
        elif provider == "openai":
            client = _TimeoutGuardedClient(OpenAILLMClient())
        elif provider == "":
            client = _TimeoutGuardedClient(LLMClient())
        else:
            client = MockLLMClient()
    except Exception as exc:
        # Loud, not silent: a missing SDK (e.g. google-generativeai not in the
        # image) or a missing key otherwise downgrades every request to the
        # mock generator with no visible sign except meta.model == "mock".
        import logging
        logging.getLogger(__name__).warning(
            "LLM_PROVIDER=%s failed to initialize (%s: %s); falling back to MockLLMClient",
            provider, type(exc).__name__, exc,
        )
        client = MockLLMClient()

    _client_cache[cache_key] = client
    return client
