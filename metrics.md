# System Performance Metrics & Evaluation Report

**Model(s):** `gemini-3.5-flash-lite` (Google, via `LLM_PROVIDER=gemini`; offline `MockLLMClient` fallback)
**Embeddings:** char n-gram (3-5, `char_wb`) TF-IDF, fit on the kit corpus (`artifacts/tfidf_vectorizer.pkl`) - no network, no fine-tuning
**Environment:** Windows laptop for the Gemini run; cache/latency benchmarks re-run on a 2-vCPU / 3 GB RAM Linux VM, Python 3.10

All numbers below are measured from `results.jsonl` (58 lines: 20 `data/input.txt` rows with their SIIS text + 38 `data/unseen_scenarios.txt` rows with no SIIS text) and `scripts/eval_paraphrase_hits.py`. Reproduce: `python scripts/generate_results.py && python scripts/validate_results.py && python scripts/eval_paraphrase_hits.py`.

## 1. Schema & Rule Compliance

| Metric | Target | Measured Value |
| :--- | :--- | :--- |
| Schema-valid output lines | >= 99% | 100% (58/58, `scripts/validate_results.py`) |
| Rule compliance (Goal / Title / Description syntax) | >= 95% | 100% (16/16 goals, 16/16 titles, 51/51 descriptions) |
| Absolute URL leaks | 0 | 0 |
| Deeplink catalog validity (exact URI match) | 100% | 100% (2/2 URIs copied verbatim from `deeplinks.json`) |
| Auto actions carrying valid actionable deeplink | >= 90% | 100% (2/2) |

Empty answers carry the brief's fallback metadata: 6 `no_match` (SIIS text present but not relevant / no extractable procedure) and 36 `no_siis_context` (unseen scenarios sent with no reference text).

## 2. Accuracy Benchmarks

| Evaluation Metric | Scale / Anchor | Score |
| :--- | :--- | :--- |
| Step accuracy (completeness, correctness, ordering) | 0.0 - 3.0 | Not independently graded: the kit has no ground-truth plans beyond `sample_output.json`, so we do not report a self-assigned score. Measured proxies over the 16 answered rows (127 steps, 51 actions): every step is checked against the SIIS text by `structure_extraction._step_is_supported` (ungrounded steps are dropped); 0 ordering violations (critical actions are last in 16/16 goals); 100% schema/rule compliance (sec. 1). |
| Deeplink relevance (exact target screen vs. parent menu) | 0.0 - 2.0 | Not independently graded. Action mix: 36 `manual`, 13 `critical`, 2 `auto` - the kit's SIIS articles for these complaints are mostly physical/service procedures, so most actions correctly carry no deeplink. Both `auto` actions got a real catalog URI (2/2). With N=2 this is not a statistically meaningful relevance score. |

## 3. Latency Benchmarks (N >= 3)

| Execution Path | Target (P95) | P50 (ms) | P95 (ms) |
| :--- | :--- | :--- | :--- |
| Cache hit - exact query match | <= 300 ms | ~2 | ~3 |
| Cache hit - unseen semantic paraphrase | <= 300 ms | ~2 | 2.5 (N=19) |
| Cold query - full pipeline extraction & mapping | <= 8000 ms | 2005 | 2515 (N=8: the `input.txt` rows that ran Stage 1 + 2 on Gemini and returned a plan; `meta.latency_ms` in `results.jsonl`. With N=8 the P95 is the slowest run.) |

Latency above is warm-process latency. It does not include a free-tier host waking from idle (e.g. the first request after the Render instance spins down); that wake-up delay is infrastructure, not query latency, and we have not benchmarked it.

## 4. Operational Cost & Cache Efficacy

| Metric Item | Target | Measured Value |
| :--- | :--- | :--- |
| Cold query average inference cost | Tracked | $0.00038 per full-pipeline cold query (mean of `meta.cost_usd` over the same 8 rows; the 5 unseen-scenario rows that only made the cheaper Stage 0 classification call average $0.00010). Derived in `llm_client.py` from the token usage the provider returns x published `gemini-3.5-flash-lite` rates ($0.10 per 1M input tokens, $0.40 per 1M output tokens). On the free tier nothing is billed; this is the paid-tier equivalent. Raw token counts are not stored in `results.jsonl`, only the resulting cost. |
| Cache hit inference cost | $0.00 | $0.00 (no Stage 1/2 call; keyword-matched Stage 0 makes no LLM call) |
| Semantic cache hit rate (on unseen paraphrases) | >= 80% | 100% (19/19); 89.5% (17/19) before the symptom vocabulary was extended after seeing the two misses - cache pre-warmed from `results.jsonl`, 20 hand-written paraphrases of the answerable input complaints sent WITHOUT `siis_response` (`scripts/eval_paraphrase_hits.py`; small, team-written set - treat as indicative) |
| Cost derivation method | - | (prompt tokens + completion tokens) x rate |

## 5. Architectural Ablation Analysis

| Architecture Variant | Step Accuracy | Latency (P95) | Cost / Query | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| Baseline: Full LLM Deeplink Mapping | not run | - | - | Not built: the brief forbids altering URIs, and a lexical retriever guarantees verbatim catalog URIs. |
| Variant A: Hybrid BM25 + Dense Embedding Retrieval | not run | - | - | Dense models blocked (huggingface.co unreachable on our network); char-n-gram TF-IDF used instead. |
| Variant B (shipped): TF-IDF retrieval + rule-based category guard | see sec. 1-2 | ~3 ms mapping | $0 | Regex guard stops restart/hardware steps getting settings deeplinks; low-similarity auto steps fall back to `bixby://dummy_positive`. |

## 6. Known Edge Cases & System Limitations

* Symptom taxonomy is keyword-based: complaints outside it are `unclassified` and never cached (always cold path).
* Bare model shorthand without "Galaxy"/"Samsung" (e.g. "my s22") is treated as an unknown device; the cache treats unknown as a wildcard, so it can still hit.
* Stage 1 quality depends on the SIIS article: when the article is off-topic for the complaint the relevance gate returns `no_match` rather than guessing.
* Misspellings: the symptom taxonomy is keyword-based, so heavy typos ("scren is blak", "screeen flikkering") come back `unclassified_issue`, are never cached and always take the cold path. Light typos that still contain a keyword are matched.
* Non-English input (e.g. Hindi) is not supported: it comes back `unclassified_issue`.
* Multi-intent complaints ("flickers, battery drains and wifi disconnects") are mapped to a single symptom category; the other symptoms are not planned for.
* The deeplink similarity cutoff (`SIMILARITY_THRESHOLD = 0.35`) is a hand-set value checked against only a handful of auto actions, not a tuned one. A regex guard (`_infer_non_auto_category`) forces restart/hardware steps to `critical`/`manual` before retrieval; it is a heuristic and can misclassify unusual phrasings.
* Deeplink retrieval is lexical; paraphrased settings names with no shared character n-grams can fall back to `dummy_positive`.

---

---

The earlier Member-A-only benchmark (Stage 0 + Stage 3 development log) is kept for reference in `docs/DEV_NOTES.md`; it is superseded by the numbers above.
