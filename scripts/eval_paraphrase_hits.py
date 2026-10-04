"""
Semantic-paraphrase cache hit rate, measured the way the brief's §6 KPI reads:
the cache is pre-warmed from results.jsonl (exactly what api.py does at
startup), then UNSEEN paraphrases of the 20 reference complaints are sent
WITHOUT a siis_response, and we count how many are served from the cache.

Only paraphrases whose source complaint actually has a non-empty answer in
results.jsonl are scored (a paraphrase of a no_match row has nothing to hit).
No LLM is called: Stage 1 is stubbed to return nothing, so any non-empty
answer below can only have come from the cache.

Run:  python scripts/eval_paraphrase_hits.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["LLM_PROVIDER"] = "mock"

from cache import SemanticCache  # noqa: E402
from deeplink_mapping import deeplink_mapping  # noqa: E402
from embeddings import TfidfEmbedder  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from schema import ContextDeeplinkResponse  # noqa: E402

# (paraphrase, 0-based row in data/input.txt it paraphrases). Written against
# the rebranded kit data (TechCorp / Nexa), mixing formal, casual, keyword,
# frustrated and typo registers, and with/without the model name.
PARAPHRASES = [
    ("My Nexa X1 display goes white and blank whenever I open certain apps.", 1),
    ("nexa x1 screen blank no text showing in apps", 1),
    ("Nexa Fold X1 screen is totally black and I can't transfer my data off it", 2),
    ("fold x1 black screen cant use data transfer!!", 2),
    ("Nexa A15 screen went black by itself after a month, shows nothing", 3),
    ("The inner display on my Nexa Fold X1 has stopped working, cover screen is fine.", 7),
    ("nexa fold x1 inner screen dead, outer one works", 7),
    ("My Nexa Fold X1 screen keeps flickering and then goes blank when I unfold it", 8),
    ("half of my nexa fold x1 screen is black, other half works", 9),
    ("How do I get rid of the floating circle shortcut on my Nexa X1 screen?", 10),
    ("Nexa X1 screen stays blank after the carrier deactivated my old phone, no activation msg", 11),
    ("my phone screen is completly cracked and unusable", 12),
    ("Nexa X1 Ultra stuck on a blue screen with tiny text and won't boot", 13),
    ("TechCorp X1 Ultra display flickers really fast every time I plug in the charger", 14),
    ("nexa x1 screen just dark, nothing visible, can't transfer data", 15),
    ("Nexa Fold X1 screen cracked at the fold and touch not working in places", 16),
    ("Nexa X1 touch input is laggy and delayed", 18),
    ("ugh my nexa x1 touchscreen lags every time i tap something", 18),
    ("Nexa X1 Ultra screen is black but the phone still rings and works", 19),
    ("x1 ultra black screen of death even though it powers on", 19),
]


def main() -> dict:
    rows = [json.loads(l) for l in (ROOT / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable = {i for i, r in enumerate(rows[:20]) if r["response"].get("contexts")}
    pipe = Pipeline(
        cache=SemanticCache(TfidfEmbedder.load()),
        stage1_fn=lambda s, e: ContextDeeplinkResponse(contexts=[]),
        stage2_fn=deeplink_mapping,
    )
    warmed = pipe.warm_from_results(ROOT / "results.jsonl")
    scored = hits = 0
    lat = []
    detail = []
    for q, src in PARAPHRASES:
        t0 = time.perf_counter()
        r = pipe.run(q, {})
        ms = (time.perf_counter() - t0) * 1000
        if src not in answerable:
            detail.append({"query": q, "source_row": src, "scored": False,
                           "reason": "source row has no answer in results.jsonl (no_match)"})
            continue
        scored += 1
        hit = r["meta"]["cache_hit"]
        hits += hit
        if hit:
            lat.append(ms)
        detail.append({"query": q, "source_row": src, "scored": True, "hit": hit,
                       "device": r["enrichment"]["device"], "symptom": r["enrichment"]["symptom_category"],
                       "similarity": r["meta"]["similarity"], "latency_ms": round(ms, 2)})
    lat.sort()
    out = {
        "warmed_entries": warmed,
        "scored": scored,
        "hits": hits,
        "hit_rate": round(hits / scored, 4) if scored else None,
        "hit_latency_p50_ms": round(statistics.median(lat), 2) if lat else None,
        "hit_latency_p95_ms": round(lat[max(0, int(0.95 * len(lat)) - 1)], 2) if lat else None,
        "detail": detail,
    }
    (ROOT / "artifacts" / "paraphrase_hit_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=2))
    for d in detail:
        if d.get("scored") and not d["hit"]:
            print("MISS:", d["query"], "|", d["device"], "|", d["symptom"])
    return out


if __name__ == "__main__":
    main()
