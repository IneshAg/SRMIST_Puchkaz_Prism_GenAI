"""
Hybrid mode: run locally, borrow the hosted Gemini key.

When this process has no LLM key of its own (get_llm_client() fell back to
the offline mock) and the operator didn't explicitly ask for the mock
(LLM_PROVIDER=mock), cache MISSES are forwarded to the team's hosted
deployment, which holds the Gemini key as a server-side secret. Stage 0 and
the semantic cache still run locally, and the forwarded answer is cached
locally, so repeats stay local and fast.

    REMOTE_API_URL   default: the team's Render deployment; set to "" to disable
    REMOTE_TIMEOUT_SECONDS   default 60 (Render's free tier can take ~50 s to wake)

Only the public /v1/troubleshoot endpoint is called -- the key never leaves
the server. A forwarded request carries FORWARD_HEADER, and the API never
forwards a request that already has it (no loops). If the remote is
unreachable, the pipeline falls back to the local mock: never a crash.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Optional

DEFAULT_REMOTE_API_URL = "https://srmist-puchkaz-theme2.onrender.com"
FORWARD_HEADER = "X-Forwarded-By-Hybrid-Client"

logger = logging.getLogger(__name__)


def remote_url() -> Optional[str]:
    url = os.getenv("REMOTE_API_URL", DEFAULT_REMOTE_API_URL).strip()
    if not url or os.getenv("RENDER"):  # never forward from the hosted service itself
        return None
    return url.rstrip("/")


def should_forward(client) -> bool:
    from llm_client import MockLLMClient

    if not isinstance(client, MockLLMClient):
        return False  # a real local key wins
    if os.getenv("LLM_PROVIDER", "").strip().lower() == "mock":
        return False  # mock explicitly requested
    return remote_url() is not None


def call_remote(query: str, siis_response) -> Optional[dict]:
    """POST to the hosted /v1/troubleshoot. Returns the parsed JSON body or
    None on any failure (logged)."""
    url = remote_url()
    if not url:
        return None
    body = {"query": query}
    if siis_response:
        body["siis_response"] = siis_response
    req = urllib.request.Request(
        f"{url}/v1/troubleshoot",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", FORWARD_HEADER: "1"},
        method="POST",
    )
    timeout = float(os.getenv("REMOTE_TIMEOUT_SECONDS", "60"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, dict) or "response" not in data:
            raise ValueError("unexpected response shape")
        return data
    except Exception as exc:
        logger.warning("Remote %s unavailable (%s: %s); using local mock", url, type(exc).__name__, exc)
        return None
