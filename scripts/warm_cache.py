"""
Bulk cache warmer + stress test.

Generates a large set of realistic complaints (every symptom the kit has a
reference article for x every device x many phrasings), runs each through the
real pipeline, and saves every NEW answer to artifacts/warm_cache.jsonl. The
API loads that file at startup (next to results.jsonl), so the answers survive
restarts and ship with the repo / Docker image / Render deploy.

Why each case is sent WITH its reference article: without reference text the
engine can only answer from cache ("no_siis_context"), so Gemini would have
nothing to extract a plan from.

Which LLM answers the misses:
  * .env has a key          -> your local Gemini
  * no key (default)        -> hybrid: forwarded to the Render deployment
  * LLM_PROVIDER=mock        -> offline mock (answers are NOT saved unless
                                 --allow-mock, so mock text never pollutes the cache)

Most cases are cache hits (that is the point of a semantic cache): only the
first phrasing per (symptom, device, article) needs an LLM call; the rest
measure the hit rate. Re-running is safe: already-saved answers are loaded
first, so it resumes where it stopped.

Usage:
    python scripts/warm_cache.py --dry-run          # just list/count the cases
    python scripts/warm_cache.py                    # run everything
    python scripts/warm_cache.py --limit 200 --rpm 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("REMOTE_TIMEOUT_SECONDS", "90")  # Render free tier can take ~50 s to wake

from cache import SemanticCache  # noqa: E402
from embeddings import TfidfEmbedder  # noqa: E402
from enrichment import SYMPTOM_TAXONOMY, enrich, extract_device  # noqa: E402
from llm_client import MockLLMClient, get_llm_client  # noqa: E402
from pipeline import Pipeline, cache_eligible, siis_fingerprint  # noqa: E402
import remote_client  # noqa: E402

OUT_DEFAULT = ROOT / "artifacts" / "warm_cache.jsonl"
_MOCK = MockLLMClient()

# Phrasing templates across the brief's registers (formal, casual, keyword,
# frustrated, typo-ish). {d} = "my <device>" / "my phone", {c} = component, {p} = problem.
TEMPLATES = [
    "{D} {c} {p}",
    "My {dn} {c} {p}, how do I fix it?",
    "{dn} {c} {p}",
    "{dn} {c} {p} help",
    "why is my {dn} {c} {p}??",
    "ugh my {dn} {c} {p} again",
    "I am facing an issue: the {c} on my {dn} {p}.",
    "{c} {p} on {dn}",
    "pls help {dn} {c} {p}",
]


def _load_kit():
    lines = [l.strip() for l in (ROOT / "data" / "input.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    siis = json.loads((ROOT / "data" / "siis_responses.json").read_text(encoding="utf-8"))["responses"]
    results = [json.loads(l) for l in (ROOT / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines, siis, results


def build_cases(per_category: int) -> list[dict]:
    """(query, siis_response, expected category). Only symptoms for which the
    kit has a reference article that produced a real answer are generated --
    for anything else Gemini has no source text and the result is empty."""
    lines, siis, results = _load_kit()
    article_for = {}            # category -> siis payload
    devices = {None}
    for i, q in enumerate(lines):
        dev = extract_device(q)
        if dev != "TechCorp device":
            devices.add(dev)
        if i < len(siis) and i < len(results) and results[i]["response"].get("contexts"):
            cat = enrich(q, llm_client=_MOCK).symptom_category
            article_for.setdefault(cat, siis[i]["siis_response"])
    devices = sorted(devices, key=lambda d: (d is None, d or ""))

    cases, seen = [], set()

    def add(q, payload, cat):
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            cases.append({"query": q, "siis_response": payload, "category": cat})

    # 1. the kit's own complaints, verbatim
    for i, q in enumerate(lines):
        if i < len(siis) and i < len(results) and results[i]["response"].get("contexts"):
            add(q, siis[i]["siis_response"], enrich(q, llm_client=_MOCK).symptom_category)

    # 2. combinatorial phrasings: symptom terms x devices x templates
    by_cat = {s.category: s for s in SYMPTOM_TAXONOMY}
    for cat, payload in article_for.items():
        sym = by_cat.get(cat)
        if not sym:
            continue
        comps = (sym.component_terms or ["phone"])[:3]
        probs = [p for p in sym.problem_terms if len(p) > 3][:10]
        made = 0
        for dev, comp, prob, tpl in itertools.product(devices, comps, probs, TEMPLATES):
            if made >= per_category:
                break
            dn = dev or "phone"
            q = tpl.format(D=f"My {dn}", dn=dn, c=comp, p=prob)
            add(q, payload, cat)
            made += 1
        # the taxonomy's own formal / casual / keyword wording per device
        for dev in devices:
            dn = dev or "phone"
            for text in (sym.formal, sym.casual, sym.keyword):
                add(f"My {dn} {sym.subject} {text}", payload, cat)
    return cases


def _wake_remote():
    url = remote_client.remote_url()
    if not url:
        return
    try:
        urllib.request.urlopen(f"{url}/health", timeout=90).read()
    except Exception as exc:
        print(f"  (could not wake {url}: {exc})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--per-category", type=int, default=120, help="generated phrasings per symptom (default 120)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cases (0 = all)")
    ap.add_argument("--rpm", type=float, default=10, help="max LLM calls (cache misses) per minute")
    ap.add_argument("--dry-run", action="store_true", help="only generate and classify-check the cases")
    ap.add_argument("--allow-mock", action="store_true", help="also save answers produced by the offline mock")
    args = ap.parse_args()

    cases = build_cases(args.per_category)
    if args.limit:
        cases = cases[: args.limit]

    # Generated text must classify back to the symptom it was built for; one
    # that doesn't is a taxonomy gap (reported, and it would never be cached).
    misclassified = []
    for c in cases:
        got = enrich(c["query"], llm_client=_MOCK)
        if got.symptom_category != c["category"] or not cache_eligible(got):
            misclassified.append((c["query"], c["category"], got.symptom_category))
    print(f"{len(cases)} cases across {len({c['category'] for c in cases})} symptoms; "
          f"{len(misclassified)} don't classify back to their own symptom")
    for q, want, got in misclassified[:15]:
        print(f"  TAXONOMY GAP: {q!r}: wanted {want}, got {got}")
    if args.dry_run:
        print(Counter(c["category"] for c in cases).most_common())
        return

    client = get_llm_client()
    mode = ("local " + client.model_name) if not isinstance(client, MockLLMClient) else (
        "hybrid -> " + str(remote_client.remote_url()) if remote_client.should_forward(client) else "offline mock")
    print(f"LLM for cache misses: {mode}")
    if mode.startswith("hybrid"):
        _wake_remote()

    pipe = Pipeline(cache=SemanticCache(TfidfEmbedder.load()))
    preloaded = pipe.warm_from_results(ROOT / "results.jsonl") + pipe.warm_from_results(args.out)
    print(f"cache pre-loaded with {preloaded} entries (results.jsonl + {args.out.name})\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    hit_ms, miss_ms = [], []
    per_cat = defaultdict(Counter)
    min_gap = 60.0 / args.rpm if args.rpm > 0 else 0.0
    last_call = 0.0
    with args.out.open("a", encoding="utf-8") as out:
        for n, c in enumerate(cases, 1):
            # throttle only before a request that will likely miss
            probe = enrich(c["query"], llm_client=_MOCK)
            likely_miss = pipe.cache.get(
                probe.canonical_query, [c["query"]] + probe.query_variations,
                device=probe.device if probe.device_confidence else None,
                context_key=siis_fingerprint(c["siis_response"]),
            ) is None
            if likely_miss and min_gap:
                wait = last_call + min_gap - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            t0 = time.perf_counter()
            try:
                r = pipe.run(c["query"], c["siis_response"])
            except Exception as exc:
                stats["error"] += 1
                print(f"[{n}/{len(cases)}] ERROR {type(exc).__name__}: {exc}")
                continue
            ms = (time.perf_counter() - t0) * 1000
            meta, resp = r["meta"], r["response"]
            if meta["cache_hit"]:
                stats["hit"] += 1
                per_cat[c["category"]]["hit"] += 1
                hit_ms.append(ms)
                continue
            last_call = time.monotonic()
            stats["miss"] += 1
            per_cat[c["category"]]["miss"] += 1
            miss_ms.append(ms)
            stats["cost_usd"] += float(meta.get("cost_usd") or 0)
            if not resp.get("contexts"):
                stats["empty"] += 1
                print(f"[{n}/{len(cases)}] miss, EMPTY ({resp.get('fallback')}) via {meta['model']}: {c['query'][:60]}")
                continue
            if meta["model"] == "mock" and not args.allow_mock:
                stats["mock_not_saved"] += 1
                continue
            row = dict(r)
            row["siis_fingerprint"] = siis_fingerprint(c["siis_response"])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            stats["saved"] += 1
            print(f"[{n}/{len(cases)}] miss -> saved ({meta['model']}, {ms:.0f} ms): {c['query'][:60]}")

    done = stats["hit"] + stats["miss"]
    pct = lambda xs, q: round(sorted(xs)[max(0, int(q * len(xs)) - 1)], 1) if xs else None
    print("\n" + "=" * 60)
    print(f"cases run        : {done}  (errors {stats['error']})")
    print(f"cache hits       : {stats['hit']}  ({stats['hit'] / done:.1%})" if done else "cache hits: 0")
    print(f"LLM calls (miss) : {stats['miss']}  saved {stats['saved']}, empty {stats['empty']}, "
          f"mock-not-saved {stats['mock_not_saved']}")
    print(f"hit latency      : p50 {pct(hit_ms, .5)} ms, p95 {pct(hit_ms, .95)} ms")
    print(f"miss latency     : p50 {pct(miss_ms, .5)} ms, p95 {pct(miss_ms, .95)} ms")
    print(f"LLM cost         : ${stats['cost_usd']:.5f}")
    print(f"saved to         : {args.out}  (loaded automatically by the API at startup)")
    print("per symptom (hit/miss):")
    for cat, cnt in sorted(per_cat.items()):
        print(f"  {cat:28s} {cnt['hit']:4d} / {cnt['miss']}")


if __name__ == "__main__":
    main()
