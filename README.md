# Theme 02 – Smart Guided Troubleshooting Engine

**Team SRMIST_Puchkaz** · SRMIST KTR · PRISM GenAI Hackathon 2026

Turns a vague device complaint ("screen went black and I can't move my data") into a validated, step-by-step troubleshooting plan with one-tap Settings deeplinks.
Repeat or paraphrased queries come back from a semantic cache in a few milliseconds.

## Demo Video

**Watch the full demonstration here:** [Google Drive Demo Video](https://drive.google.com/file/d/1_ooCOpjRS2G8lOtwKyPkBSGQNYcRL5gH/view?usp=sharing)

## Live demo (no setup needed)

**https://srmist-puchkaz-theme2.onrender.com** - hosted with Gemini enabled. The key is stored as a server-side secret on Render, not in this repo.

- Interactive API (try it in the browser): **https://srmist-puchkaz-theme2.onrender.com/docs**
- Health: `GET /health`

```bash
curl -X POST https://srmist-puchkaz-theme2.onrender.com/v1/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{"query": "My Nexa X1 touch is laggy and delayed"}'
```

Measured on the live service: cache hits answer in ~5 ms server-side (~250 ms round trip from India); a new complaint with reference text runs Gemini end-to-end in ~4.2 s.

> Free tier: the service sleeps after 15 min idle, so the **first request can take ~1 minute** while it wakes up. Requests after that are fast.

## Pipeline

```
complaint (+ optional SIIS reference text)
   │
   ├─ Stage 0  enrichment.py            normalize → device + symptom, 8–10 paraphrases
   ├─ Stage 3  cache.py                 semantic lookup ── hit → return (~3 ms)
   │                                                     └ miss ↓
   ├─ Stage 1  structure_extraction.py  LLM → Goal / Actions / Steps, validated + repaired
   ├─ Stage 2  deeplink_mapping.py      catalog match, category guard, critical actions last
   └─ scrubber.py → cache write → JSON response
```

| File | Role |
| --- | --- |
| `src/api.py` | FastAPI app (`/v1/troubleshoot`, `/health`) |
| `src/pipeline.py` | Wires all stages, fallback metadata, cache pre-warm |
| `src/enrichment.py` | Stage 0: device/symptom extraction, paraphrase generation |
| `src/structure_extraction.py` | Stage 1: LLM extraction + programmatic validation/repair |
| `src/relevance_gate.py` | Rejects SIIS text that doesn't match the complaint (→ `no_match`) |
| `src/deeplink_mapping.py` | Stage 2: TF-IDF catalog matching, category guard, ordering |
| `src/cache.py` | Stage 3: semantic cache (device- and source-gated) |
| `src/scrubber.py` | Removes any URL / markdown link from the output |
| `src/rule_extraction.py` | Offline Stage 1: builds the plan from the article's headings and instruction sentences when no LLM is reachable |
| `src/remote_client.py` | Hybrid mode: forwards cache misses to the hosted service when there is no local key |
| `src/llm_client.py` | Gemini / OpenAI / offline mock, timeout + cost tracking |
| `src/embeddings.py` | Char n-gram TF-IDF embedder (no network needed) |
| `src/schema.py` | Official response schema |
| `tests/` | 244 pytest tests |

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_corpus_vectorizer.py
pytest -q
cd src && uvicorn api:app --port 8000      # then open http://localhost:8000/docs
```

### Three ways it runs (hybrid)

No API key is needed to run it locally. The engine picks the best LLM it can reach:

| Mode | When | Who builds the plan |
| --- | --- | --- |
| **Local key** | `.env` has `LLM_PROVIDER=gemini` + `GOOGLE_API_KEY` | Gemini, from your machine |
| **Hybrid (default)** | No key set | Gemini on the hosted Render service: new complaints are forwarded to it, answers are cached locally |
| **Offline** | `LLM_PROVIDER=mock`, or Render unreachable | `rule_extraction.py`: no LLM, plan built from the article's own headings and instructions |

In hybrid mode Stage 0 and the semantic cache still run locally. Only cache misses go to Render's public `/v1/troubleshoot`, so the key never leaves the server. Forwarded answers show `"model": "remote:gemini-…"` in `meta`. If Render is asleep, the first forwarded request can take ~1 minute.

**Offline quality.** Without any LLM, Stage 1 reads the reference article's structure: each heading ("Step 2: Force a Restart") becomes an action, and the instruction sentences under it become its steps. Chains like "go to Settings, tap Display, and then tap…" are split one interaction per step, and informational sections are skipped. Every step is copied from the article, so nothing is invented, and the result goes through the same validation, category guard and deeplink mapping as Gemini's output. On the 20 kit complaints it answers the same 14 that Gemini does, with no URL leaks and no deeplinks outside the catalog. The wording is plainer than Gemini's: descriptions are formulaic ("It will help with safe mode") and plans keep more of the article's raw sentences.

**For judges:** the quickest and best-quality path is the live URL above, with no setup. Running locally without a key depends on the Render service being up. If it's asleep the first request waits ~1 minute; if it's unreachable, the engine falls back to offline mode rather than failing.

To use your own key instead: `cp .env.example .env` and fill in the two lines.

### Pre-warming the cache in bulk

```bash
python scripts/warm_cache.py --dry-run   # list the ~880 generated test complaints
python scripts/warm_cache.py             # run them all; new answers -> artifacts/warm_cache.jsonl
```

The script generates every combination of symptom × device × phrasing for which the kit has a reference article. It runs them through the real pipeline and saves each new Gemini answer. The API loads `artifacts/warm_cache.jsonl` at startup (next to `results.jsonl`), so the answers survive restarts and ship with the repo.

- **Few LLM calls:** most cases are cache hits, so only ~20 need Gemini (rate-limited with `--rpm`, default 10/min).
- **Resumable:** a re-run loads what's already saved and picks up where it stopped.
- **No mock answers saved:** answers from the offline mock are never written to the file.
- **Taxonomy check:** it reports any generated complaint that doesn't classify back to its own symptom.

Docker (same three modes):

```bash
docker build -t theme2 .
docker run -p 8000:8000 theme2                    # hybrid
docker run -p 8000:8000 --env-file .env theme2    # local key
```

Deploy your own copy: `render.yaml` is a Render Blueprint (Dashboard → New → Blueprint → this repo). Render asks for `GOOGLE_API_KEY` once and keeps it as a secret.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | unset (hybrid) | `gemini`, `openai`, or `mock` (offline, never forwards) |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` | – | Key for a local provider |
| `LLM_MODEL` | provider default (`gemini-3.5-flash-lite`) | Override the model |
| `REMOTE_API_URL` | the Render URL above | Where hybrid mode forwards misses; empty = disable |
| `REMOTE_TIMEOUT_SECONDS` | `60` | Timeout for a forwarded request |
| `CACHE_WARM_FILE` | `results.jsonl` + `artifacts/warm_cache.jsonl` | Files used to pre-warm the cache (`;` on Windows, `:` elsewhere); empty = start cold |
| `REQUEST_TIMEOUT_SECONDS` | `70` | Hard per-request timeout (returns 504) |

If anything fails (missing key or SDK, Render down), the API logs a warning and falls back to the offline mock instead of crashing.

## API

`POST /v1/troubleshoot`

```json
{ "query": "phone swipe gestures wrong direction after app install",
  "siis_response": { "title": "...", "content": "..." } }
```

`siis_response` is optional. It can be an object or a plain string.

Response:

```json
{ "query": "...",
  "query_variations": ["... 8–10 paraphrases ..."],
  "response": { "contexts": [ /* Goal objects per schema.py */ ],
                "fallback": "no_match | no_siis_context" },
  "meta": { "cache_hit": true, "similarity": 0.94, "latency_ms": 3.1,
            "model": "gemini-3.5-flash-lite", "cost_usd": 0.0 },
  "enrichment": { "device": "...", "symptom_category": "...", "...": "..." } }
```

`fallback` appears only when `contexts` is empty:

- `no_siis_context` means no reference text was sent.
- `no_match` means reference text was sent but held no matching procedure.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | `{"status": "ok"}` |
| `POST /v1/enrich` | Stage 0 output only |
| `GET /v1/cache/stats` | Cache size and threshold |

### Example plan (from `results.jsonl`, trimmed)

Query: *"My Nexa X1 screen inputs are delayed and the touch responsiveness is laggy…"*

```json
{ "goal": "Follow these steps to perform this Touchscreen Troubleshooting",
  "title": "Touchscreen issues",
  "score": 0.9,
  "actions": [
    { "actionName": "Touch Sensitivity Settings",
      "description": "It will adjust touch sensitivity settings.",
      "category": "auto",
      "stepGroups": [{
        "steps": ["Go to Settings", "Tap Display", "Tap the switch next to Touch sensitivity"],
        "actionableDeeplink": { "deeplink": "voiceassist://masked/act/1b0d34e9b4",
                                "message": "Disable Touch sensitivity", "originalType": "offURL" },
        "validationDeeplink": { "deeplink": "voiceassist://masked/val/6451858b28",
                                "key": "Touch sensitivity" } }] },
    { "actionName": "Factory Data Reset", "category": "critical", "...": "..." }
  ] }
```

## Results

Full report in `metrics.md`, which follows the brief's Appendix C template.

| Metric | Target | Measured |
| --- | --- | --- |
| Schema-valid lines | ≥ 99% | 100% (58/58) |
| URL leaks | 0 | 0 |
| Catalog URI validity | 100% | 100% |
| Paraphrase cache hit rate | ≥ 80% | 100% (19/19)* |
| Cache-hit P95 | ≤ 300 ms | ~3 ms |
| Cold-path P95 (Gemini) | ≤ 8 s | 2.5 s |

\* 17/19 before the symptom vocabulary was extended with real-world phrasings; the two misses ("blue screen", "screen just dark") are now covered. See `docs/DEV_NOTES.md`.

Reproduce:

```bash
python scripts/generate_results.py      # runs all 58 queries → results.jsonl
python scripts/validate_results.py      # schema + rule + URL-leak check
python scripts/eval_paraphrase_hits.py  # paraphrase hit rate on a pre-warmed cache
python scripts/benchmark_cache.py       # cache / Stage 0 latency
```

## How it meets the brief

| Brief requirement | How |
| --- | --- |
| Schema conformance | Every response is validated against `schema.py` before it is returned |
| Zero URL leaks | `scrubber.py` strips URLs and markdown links from all text fields |
| Catalog integrity | Deeplinks are copied verbatim from `deeplinks.json`; nothing is generated |
| No hallucinated steps | Every step is checked against the SIIS text; unsupported steps are dropped |
| Empty result + fallback | `contexts: []` always carries `no_match` or `no_siis_context` |
| Field rules (goal, title, 5–7 word description) | Enforced and repaired in code, not left to the prompt |
| Plan hierarchy | Settings actions first, `critical` (restart / reset / safe mode) always last |
| Semantic cache keys | Embedding similarity over the canonical query + paraphrases, not exact strings |
| Deterministic output | Temperature 0, memoized LLM calls, deterministic Stage 0 |
| Cost tracking | `meta.cost_usd` from provider token counts × published rates |
| Pure JSON | Pydantic response models; code fences stripped from LLM output |

## Key design choices

- **Deterministic Stage 0.** A keyword taxonomy (vocabulary extended with phrasings from support forums and repair guides) covers 35 symptom categories, so it can't hallucinate a symptom. The LLM is only a whitelisted fallback for complaints the keywords don't match.
- **Safe cache.**
  - A hit is never served across different devices.
  - A request that sends SIIS text only hits entries built from that same text.
  - Empty and low-confidence answers are never cached.
- **Pre-warmed.** On startup the cache loads `results.jsonl`, so requests without SIIS text still get answers.
- **TF-IDF, not a sentence-transformer.** HuggingFace is blocked on our network. Char n-grams are fast and handle typos.
- **Repair, don't reject.** Titles, goals, descriptions and action order are fixed in code, so one bad field doesn't discard a whole valid plan.
- **Category guard.** Restart, reset and safe mode are `critical`. Hardware and service-centre steps are `manual`. Neither carries a deeplink.

## Limitations

- The symptom taxonomy is keyword-based. Novel complaints are marked `unclassified` and always take the cold path.
- Deeplink retrieval is lexical. Paraphrased screen names can fall back to `voiceassist://dummy_positive`.
- The cache uses one global lock: correct under concurrency, but not tuned for throughput.
- The paraphrase hit-rate set is small (19 scored queries) and team-written, and the vocabulary was extended after seeing its misses, so treat 100% as optimistic.

Detailed design notes and the bug log are in [`docs/DEV_NOTES.md`](docs/DEV_NOTES.md).
