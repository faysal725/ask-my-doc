# Ask My Doc — Technical Documentation

Full engineering documentation for the Ask My Doc RAG system. Where the `README.md` gives a portfolio-facing overview, this document goes deeper: every component, every design decision, the full deployment path, and the real problems hit while building it.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Component Reference](#component-reference)
4. [API Reference](#api-reference)
5. [Configuration & Environment Variables](#configuration--environment-variables)
6. [Local Development Setup](#local-development-setup)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)
9. [Engineering Decisions & Tradeoffs](#engineering-decisions--tradeoffs)
10. [Troubleshooting Log](#troubleshooting-log)
11. [Known Limitations](#known-limitations)
12. [Future Work](#future-work)

---

## System Overview

Ask My Doc is a hybrid-retrieval RAG (Retrieval-Augmented Generation) system that answers questions about Next.js documentation, with every answer grounded in and citing the retrieved source material. It was built as a portfolio project to demonstrate production-oriented AI engineering practices — not a minimal tutorial pipeline.

**Core capabilities:**
- Ingest markdown/MDX documents into a searchable knowledge base
- Answer natural-language questions using only retrieved context
- Cite the specific source sections used to generate each answer
- Measure system quality with automated, LLM-judged evaluation metrics

**Constraints the system was designed under:**
- $0 budget — every service used is on a free tier
- Weak local compute — no local GPU inference; all embedding, reranking, and generation happens via hosted APIs
- Deployment targets: Render (backend) and Vercel (frontend), both free tier

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          INGESTION PATH                          │
│                                                                    │
│  Raw .mdx file                                                    │
│       │                                                            │
│       ▼                                                            │
│  chunking.py          Heading-aware split, code/table protected   │
│       │                                                            │
│       ▼                                                            │
│  embedding.py         Gemini gemini-embedding-001 (768-dim)       │
│       │                                                            │
│       ▼                                                            │
│  vector_store.py      Upsert into Qdrant (deterministic UUIDs)    │
│                                                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                           QUERY PATH                              │
│                                                                    │
│  User question                                                    │
│       │                                                            │
│       ├──────────────────┬──────────────────┐                    │
│       ▼                  ▼                                        │
│  vector_store.py    bm25_index.py                                 │
│  (dense search)     (sparse/keyword search)                       │
│       │                  │                                        │
│       └────────┬─────────┘                                        │
│                 ▼                                                  │
│         hybrid_retrieval.py                                       │
│         Reciprocal Rank Fusion (RRF, k=60)                        │
│                 │                                                  │
│                 ▼                                                  │
│           reranker.py                                             │
│           Cohere Rerank API (rerank-v3.5)                         │
│                 │                                                  │
│                 ▼                                                  │
│          llm_client.py                                            │
│          Groq (Llama 3.3 70B) — grounded, cited generation        │
│                 │                                                  │
│                 ▼                                                  │
│         Answer + cited sources → frontend                         │
│                                                                    │
└─────────────────────────────────────────────────────────────────┘
```

**Layered backend architecture** (`backend/app/`):

```
app/
├── main.py            FastAPI entry point, CORS, health check, router mounting
├── core/
│   └── config.py       Centralized Pydantic settings, loads .env
├── routers/
│   ├── ingest.py        POST /ingest — HTTP layer only
│   └── query.py         POST /query — HTTP layer only
├── services/            All business logic lives here, framework-agnostic
│   ├── chunking.py
│   ├── embedding.py
│   ├── vector_store.py
│   ├── bm25_index.py
│   ├── hybrid_retrieval.py
│   ├── reranker.py
│   └── llm_client.py
└── models/
    └── schemas.py        Pydantic request/response schemas
```

The separation between `routers/` (HTTP concerns) and `services/` (logic) means every service function is independently testable and callable outside of FastAPI — which is exactly how the evaluation suite uses them (imports `hybrid_search`, `rerank`, and `generate_answer` directly, bypassing HTTP entirely).

---

## Component Reference

### `chunking.py` — Heading-aware markdown chunking

Splits a raw markdown/MDX document into retrieval-ready chunks.

**Algorithm:**
1. Strip YAML frontmatter (`--- ... ---`), extract `title` if present.
2. Walk the document line by line, tracking a heading stack (`[(level, text), ...]`) that gives every chunk a full heading path, e.g. `"Prefetching > Controlling prefetching > Manual prefetch"`.
3. Detect fenced code blocks (line starting with `` ``` ``, regardless of language tag or extra attributes like `filename="x"`). While inside a fence, all other parsing (including heading detection) is suspended — this prevents a `#` inside a code comment, or a pipe character in a TypeScript union type, from being misread as markdown structure.
4. Detect markdown tables (`|`-prefixed lines) and keep them atomic.
5. Close a chunk whenever a new heading is hit, or at end of document.
6. Post-process: any chunk exceeding ~500 tokens is split further at paragraph (`\n\n`) boundaries via `_split_oversized()` — but a paragraph split is only accepted if it doesn't fall inside a code fence (re-scanned before splitting), and an oversized code block that can't be shrunk without breaking is left whole rather than corrupted.

**Output shape** — each chunk is a dict:
```python
{
    "text": str,
    "heading_path": str,
    "token_count": int,
    "contains_code": bool,
    "chunk_index": int,       # sequential, unique within a single document
    "source_doc": str,
}
```

**Design decision — no chunk overlap.** Unlike naive fixed-size chunking (which typically uses ~10-15% overlap to compensate for arbitrary cut points), heading-aware chunking already preserves section context via the heading path metadata. Overlap was judged unnecessary and was deliberately not implemented.

**Token counting** uses `tiktoken`'s `cl100k_base` encoding as an approximation. This is OpenAI's tokenizer, not Llama's — token counts are close enough for chunk-size budgeting purposes, but not exact.

### `embedding.py` — Async Gemini embeddings

Wraps Google's Gemini embedding API (`gemini-embedding-001`, 768 dimensions via Matryoshka truncation).

**Critical constraint discovered during development:** Gemini's embedding API accepts only one input text per request — unlike Cohere/OpenAI-style array batching. The batching strategy compensates with `asyncio.Semaphore(5)` concurrency control plus logical batch checkpointing (groups of 50), rather than true array batching.

**Two distinct public functions:**
- `embed_documents(chunks: list[dict]) -> tuple[list[dict], list[dict]]` — batch embeds chunk text for ingestion, uses `task_type="RETRIEVAL_DOCUMENT"`. Returns `(successful, failed)` — failures are isolated per-chunk, not fatal to the whole batch.
- `embed_query(text: str) -> list[float]` — embeds a single search query, uses `task_type="RETRIEVAL_QUERY"`. Using a distinct task type for queries vs. documents is a Gemini-specific optimization that improves retrieval quality.

**Error handling:** retry with exponential backoff (`2^attempt + jitter`, capped at 30s) across three cases — `ClientError` with `code == 429` (rate limit), `ServerError` (5xx), and a catch-all `Exception` for transport-level failures (network drops, timeouts) that aren't covered by the SDK's typed exceptions. After `MAX_RETRIES` (5), a chunk is marked failed rather than crashing the whole ingest run.

**Composite chunk IDs:** since `chunk_index` resets to 0 for every document, a naive lookup dict keyed only by `chunk_index` would collide across a multi-document ingestion batch. The fix: build lookup keys as `f"{source_doc}::{chunk_index}"` throughout the embedding pipeline.

**Why `async def`, not sync wrapped in `asyncio.run()`:** FastAPI route handlers run inside an already-active event loop; calling `asyncio.run()` from within one raises `RuntimeError`. Both `embed_documents` and `embed_query` are natively `async` so they can be `await`ed directly from async routes.

### `vector_store.py` — Qdrant wrapper

Manages the Qdrant collection: creation, point construction, upsert, retrieval, and search.

**Collection setup** — `ensure_collection()` is idempotent (checks `collection_exists()` before creating), called defensively at the top of `upsert_chunks()` so call order never matters. Collection name: `nextjs_docs_v1` (versioned naming, allows rebuilding a v2 collection later without touching v1).

**Point ID scheme — deterministic UUIDs.** Each chunk's Qdrant point ID is `uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_doc}::{chunk_index}")`. This makes ingestion idempotent: re-running ingest on the same document overwrites the existing points (same deterministic ID) rather than creating duplicates — essential during iterative development, where re-ingesting the same test file happens constantly.

**Payload** stores every chunk field except the raw vector itself, plus an explicit `chunk_id` field (the same composite key used for the UUID) so both dense and sparse search results carry a consistent identifier for later fusion.

**`search_vectors(query_vector, top_k)`** — dense similarity search via `client.query_points()`. Wrapped in try/except that logs the error and returns `[]` rather than propagating — a search failure degrades to "no dense results" rather than crashing the whole hybrid search.

**`fetch_all_chunks()`** — paginates through the entire collection via `client.scroll()` (loop until `next_offset` is `None`), used to rebuild the BM25 index from Qdrant as the single source of truth, rather than maintaining a second, separately-synced chunk list.

**Reliability additions:**
- `_build_point()` raises a clear `ValueError` if a chunk is missing its `vector` field, rather than letting a malformed point fail deep inside the Qdrant client with an opaque error.
- `upsert_chunks()` retries each batch up to 2 times with a 1-second delay on failure, isolating transient network blips from a full ingest failure.

### `bm25_index.py` — Keyword search

Classic BM25 (Okapi) scoring over the full chunk corpus, using the `rank-bm25` library. Runs entirely in memory — no external service, since keyword search over a documentation-sized corpus is cheap.

**Why BM25 matters alongside vector search:** dense embeddings are excellent at capturing meaning but notoriously weak at exact-term matching — a query containing a specific function name (`revalidatePath`), config key, or error code can be outscored by semantically-similar-but-wrong chunks. BM25 catches exactly this failure mode.

**Tokenization:** `re.findall(r"\w+", text.lower())` — simple, no stopword removal (BM25's term-frequency weighting handles common words reasonably well without it).

**Caching:** `get_or_build_index()` builds the index once per process lifetime (lazy, cached in module-level state), pulling chunks via `vector_store.fetch_all_chunks()`. `refresh_index()` forces a rebuild — intended to be called after ingestion so newly-added documents become searchable without an app restart, though this call currently must be triggered manually rather than happening automatically on every ingest.

### `hybrid_retrieval.py` — Reciprocal Rank Fusion

Combines dense (vector) and sparse (BM25) search results into a single ranked list.

**Why not just average the raw scores:** cosine similarity from Qdrant (~0–1 range) and BM25 scores (unbounded, depends on term frequency and document length) live on incompatible scales. Averaging them would be meaningless — a BM25 score of 8.2 isn't "more relevant" than a cosine score of 0.8 in any comparable sense.

**Reciprocal Rank Fusion (RRF)** sidesteps this by fusing *rank positions*, not raw scores:

```
RRF_score(chunk) = Σ 1 / (k + rank_in_list)
```

summed across every list the chunk appears in (rank is 1-indexed; `k=60` is the standard damping constant). A chunk appearing in only one of the two result lists simply doesn't contribute a term from the list it's absent from — no special-case logic needed to handle that scenario.

**Implementation steps** (`hybrid_search()`):
1. Embed the query, run dense search (`top_k_dense`, default 30) and sparse search (`top_k_sparse`, default 30) independently.
2. Build `chunk_id → rank` dictionaries for each result list.
3. Union all chunk IDs seen in either list.
4. Compute RRF score per chunk ID, using the formula above.
5. Sort descending by RRF score, return top `final_k` (default 10) with full chunk data reattached.

### `reranker.py` — Cohere Rerank API

Second-stage precision reranking, run only on the hybrid retrieval shortlist (not the full corpus).

**Why a second stage at all:** the embedding model used for retrieval is a *bi-encoder* — it embeds the query and each chunk independently, then compares vectors. This is fast (embed once, compare against thousands of pre-computed vectors) but loses information, since the model never sees the query and chunk together. A *cross-encoder*-style reranker (which Cohere's Rerank API implements as a hosted service) scores the query and each candidate jointly, catching relevance signals bi-encoder similarity alone misses — at the cost of being too slow to run against the full corpus, hence running it only on the already-narrowed candidate set.

**Implementation history — originally self-hosted, later moved to a hosted API.** The reranker was originally implemented with a local `sentence-transformers` cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which worked correctly in local development. It was replaced with Cohere's hosted Rerank API (`rerank-v3.5`) after deployment revealed that loading `torch` + `transformers` + the cross-encoder model exceeded Render's 512MB free-tier memory limit before the app could even bind to a port. Switching to a hosted reranking API removed the entire `torch`/`sentence-transformers` dependency tree from the backend, fixing the memory crash and — more importantly — bringing the reranking stage into line with the project's own stated principle (weak local compute → hosted APIs for everything), which the self-hosted cross-encoder had quietly violated.

**Current implementation:**
```python
response = client.rerank(model="rerank-v3.5", query=query, documents=texts, top_n=top_k)
```
Each result's `relevance_score` (already 0–1 normalized) is attached to the corresponding chunk as `rerank_score`.

### `llm_client.py` — Groq generation

Generates the final grounded answer using Groq-hosted Llama 3.3 70B (default), with the model name overridable per call (used by the evaluation suite to substitute a smaller model with a separate rate-limit bucket).

**System prompt** constrains the model to answer only from provided context, explicitly instructs it to state when context is insufficient rather than fabricate an answer, and requires citing source chunks by their `heading_path`.

**Context construction** (`build_context()`) numbers each chunk (`[Source 1: ...]`, `[Source 2: ...]`) so the model has stable references to cite by.

**Error handling:** retries on `RateLimitError` and `APIConnectionError` with exponential backoff; `APIError` (auth, malformed request) is treated as non-retryable and re-raised immediately as a `RuntimeError`. An empty chunk list short-circuits to a static "insufficient context" response rather than sending an empty prompt to the LLM.

---

## API Reference

### `POST /ingest`

**Request:**
```json
{ "filename": "example.mdx", "content": "raw markdown/mdx text..." }
```

**Response (200):**
```json
{
  "filename": "example.mdx",
  "chunks": 11,
  "embedded": 11,
  "upserted": 11,
  "errors": []
}
```

**Error responses:**
| Code | Cause |
|---|---|
| 400 | Empty `content` or missing `filename` |
| 422 | Chunking produced zero chunks (malformed content), or all chunks failed to embed |
| 502 | Upstream failure (Gemini embedding, Qdrant upsert) |

### `POST /query`

**Request:**
```json
{ "query": "how does prefetching work in next.js", "top_k": 5 }
```

**Response (200):**
```json
{
  "answer": "Prefetching in Next.js is... (Source: Prefetching > How does prefetching work?)",
  "sources": [
    {
      "heading_path": "Prefetching > How does prefetching work?",
      "source_doc": "3.mdx",
      "text": "When navigating between routes...",
      "rerank_score": 0.916
    }
  ]
}
```

If retrieval returns no results, the response degrades gracefully to an explicit "I couldn't find any relevant information" answer with an empty `sources` array, rather than erroring.

**Error responses:**
| Code | Cause |
|---|---|
| 400 | Empty `query` |
| 502 | Retrieval, reranking, or generation failure (each stage wrapped independently, so the error message identifies exactly which stage failed) |

### `GET /health`

Returns `{"status": "ok"}`. Used for manual sanity checks and by Render to determine service liveness.

---

## Configuration & Environment Variables

All configuration is centralized in `app/core/config.py` via a Pydantic `Settings` class, loaded from a `.env` file whose path is resolved relative to `config.py`'s own location (not the process's current working directory) — this matters because the evaluation suite runs from a different directory (`eval/`) and would otherwise fail to find `backend/.env`.

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | LLM generation (Groq, Llama 3.3 70B) |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `GEMINI_API_KEY` | Embeddings (Gemini, gemini-embedding-001) |
| `COHERE_API_KEY` | Reranking (Cohere Rerank v3.5) |
| `CORS_ORIGINS` | Allowed frontend origin(s) for CORS |
| `APP_ENV` | `development` / `production` |

Frontend (`.env.local`):

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_API_URL` | Backend base URL the frontend calls |

---

## Local Development Setup

**Backend:**
```bash
cd backend
poetry install
cp .env.example .env   # fill in real API keys
poetry run uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` for interactive API testing.

**Frontend:**
```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```
Visit `http://localhost:3000`.

Both must run simultaneously for the frontend to function — the backend's CORS configuration must include the frontend's origin.

---

## Evaluation

**Methodology:** 20 hand-written question/ground-truth pairs covering the three ingested documentation pages (Prefetching, ISR/Cache Components, Babel configuration). The domain was chosen deliberately — since the author has direct expertise in Next.js, the hand-written ground truths are trustworthy rather than guessed at, which matters because RAGAS's reference-based metrics are only as good as the reference answers.

**Metrics** (via RAGAS, LLM-judged):

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.88 | Are generated claims actually supported by retrieved context (hallucination check) |
| Answer Relevancy | 0.82 | Does the answer address the question actually asked |
| Context Recall | 0.96 | Did retrieval find all information needed to answer correctly |
| Context Precision | 0.88 | Are the retrieved chunks actually relevant / well-ranked |

**Why evaluation runs in a separate Python 3.12 environment.** The main backend targets Python 3.14. RAGAS's dependency tree (specifically `scikit-network` and `dill`, used internally for content-hashing and knowledge-graph testset generation) had unresolved build/compatibility failures on 3.14 at the time of writing — `scikit-network` had no prebuilt wheel and failed to compile from source without a full C++ toolchain, and `dill`'s pickling internals were incompatible with changes in Python 3.14's `pickle` module. Rather than downgrade the entire production backend to chase RAGAS compatibility, the evaluation suite (`eval/`) runs in its own isolated `eval/.venv` on Python 3.12, importing the backend's service modules directly via a `sys.path` insert. This keeps the production app on the newer Python version while unblocking evaluation — a deliberate, documented tradeoff rather than an oversight.

**Running the evaluation:**
```bash
cd eval
python -m venv .venv
source .venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install ragas langchain-openai langchain-google-genai groq qdrant-client python-dotenv pydantic-settings google-genai tiktoken rank-bm25 cohere
python run_ragas_eval.py
```

**Evaluator model:** Groq's `llama-3.1-8b-instant` was used as the LLM judge rather than the production `openai/gpt-oss-120b`, specifically to draw from a separate daily token-quota bucket — the 70B model's free-tier quota (100K tokens/day) was repeatedly exhausted during iterative debugging of the eval pipeline itself, while the 8B model's quota (500K tokens/day) was untouched.

**Concurrency note:** RAGAS's default evaluation concurrency exceeded Groq's free-tier rate limit (30 requests/minute on the 8B model) when running faithfulness and context-recall checks, which each require multiple sequential LLM calls per sample. This caused widespread `TimeoutError`s. Fixed by explicitly passing a `RunConfig(max_workers=3, timeout=120)` to `evaluate()`, trading run duration for reliability.

---

## Deployment

### Backend (Render)

- Root directory: `backend`
- Build command: `poetry install --no-root`
- Start command: `poetry run uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Instance type: Free (512MB RAM)
- Environment variables: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `GEMINI_API_KEY`, `COHERE_API_KEY`, `CORS_ORIGINS`, `APP_ENV`, `POETRY_VERSION`

**`POETRY_VERSION` is required** — Render's default Poetry version (2.1.3 at time of deployment) was older than the local development Poetry version (2.4.1) used to generate `poetry.lock`, and the two versions disagreed on whether the lock file was valid, causing every build to fail with "pyproject.toml changed significantly since poetry.lock was last generated" despite the lock file being genuinely in sync locally (`poetry check` passed). Setting `POETRY_VERSION=2.4.1` as an environment variable resolved it.

**Free tier note:** the instance spins down after inactivity; the first request after idle can take 30–60 seconds while the container cold-starts.

### Frontend (Vercel)

- Root directory: `frontend`
- Framework: Next.js (auto-detected)
- Environment variable: `NEXT_PUBLIC_API_URL` pointing at the deployed Render backend URL
- After deployment, the backend's `CORS_ORIGINS` must be updated to match the exact Vercel URL (no trailing slash) or all frontend requests will fail with CORS errors.

---

## Engineering Decisions & Tradeoffs

A summary of the nontrivial decisions made during development, with the reasoning behind each:

1. **Heading-aware chunking over fixed-size chunking.** Documentation mixes prose, code, and tables; naive fixed-character chunking would routinely slice code examples in half. A custom chunker was built that treats code fences and tables as atomic units and uses markdown headings as natural chunk boundaries.

2. **No chunk overlap.** Justified by heading-aware chunking already preserving section context — overlap exists in naive chunking specifically to compensate for arbitrary cut points, which don't apply here.

3. **RRF over raw score combination for hybrid retrieval.** Cosine similarity and BM25 scores are on incompatible scales; combining rank positions instead of raw scores is the standard, correct approach.

4. **Two-stage retrieval (broad hybrid search → precise reranking).** Balances retrieval speed (bi-encoder embeddings compare fast) against precision (cross-encoder-style reranking is accurate but too slow to run against the full corpus) by only reranking the already-narrowed shortlist.

5. **Deterministic UUIDs for vector store point IDs.** Makes re-ingestion idempotent — critical during iterative development, where the same test document gets re-ingested repeatedly.

6. **Composite chunk IDs (`source_doc::chunk_index`).** Prevents ID collisions when multiple documents are ingested together, since `chunk_index` alone resets to 0 per document.

7. **Reranker moved from self-hosted cross-encoder to hosted Cohere API.** Originally implemented with local `sentence-transformers`, which worked in development but exceeded Render's free-tier memory limit at deploy time. Switching to a hosted API both fixed the deployment crash and brought the component into alignment with the project's own "no local heavy compute" principle.

8. **Evaluation suite isolated in its own Python 3.12 virtual environment.** RAGAS's dependency tree had unresolved compatibility issues with Python 3.14 (the version the main backend targets). Rather than downgrade the production app, evaluation was isolated into its own environment.

9. **CPU-only PyTorch wheel source pinned in `pyproject.toml`** (relevant during the self-hosted-reranker phase, retained as a documented lesson even after the reranker was replaced) — the default PyPI PyTorch wheel pulled in a full CUDA/GPU dependency chain (`nvidia-cublas`, `nvidia-cudnn`, etc.), which is both unnecessary on a CPU-only deployment target and a significant contributor to the memory/disk footprint that caused the original deployment failure.

---

## Troubleshooting Log

A record of real problems hit during development, kept because the resolutions are non-obvious and likely to recur:

| Problem | Root Cause | Resolution |
|---|---|---|
| `poetry env` silently switched from Python 3.14 to 3.12 | Installing Python 3.12 (for the eval environment) added it to PATH; Poetry picked it up as the new default for a `requires-python = ">=3.11,<4.0"` project | Not a bug — both versions satisfy the constraint; verified the new environment had all dependencies working correctly and continued with it |
| `scikit-network` failed to build on Python 3.14 | No prebuilt wheel for 3.14 on Windows; building from source requires a C/C++ toolchain not installed | Pinned an older RAGAS version initially, then ultimately isolated evaluation into a separate Python 3.12 environment where prebuilt wheels exist |
| `dill`/`datasets` `TypeError: Pickler._batch_setitems()` | `dill`'s pickling internals incompatible with Python 3.14's changed `pickle` module | Same fix — isolated Python 3.12 environment for evaluation |
| Render build failing with "pyproject.toml changed significantly since poetry.lock was last generated" despite `poetry check` passing locally | Local Poetry version (2.4.1) and Render's default Poetry version (2.1.3) disagreed on lock file validity | Set `POETRY_VERSION=2.4.1` as a Render environment variable |
| Render deploy crashing with "Out of memory (used over 512Mi)" before binding a port | Self-hosted cross-encoder reranker pulled in `torch` + `transformers` + `sentence-transformers`, whose baseline import/load memory exceeded the free-tier 512MB limit | Replaced the self-hosted reranker with the hosted Cohere Rerank API, removing the entire heavy dependency tree |
| `ragas` embeddings call failing with 404 on `models/embedding-001` and later `models/text-embedding-004` | Gemini embedding model names change/deprecate; the correct current model for the LangChain wrapper is `models/gemini-embedding-001` | Used the model name confirmed from current LangChain documentation rather than an assumed/deprecated one |
| RAGAS evaluator LLM calls failing with `'n' : number must be at most 1` | `ResponseRelevancy`'s default self-consistency check requests multiple generations (`n>1`) in one call; Groq's API doesn't support `n>1` (unlike OpenAI's) | Instantiated `ResponseRelevancy(strictness=1)` to limit it to a single generation |
| Groq daily token quota (100K TPD) exhausted mid-evaluation | Free tier quota shared across all testing that day (ingestion, querying, and repeated failed eval runs) | Switched the evaluator LLM to `llama-3.1-8b-instant`, which has its own separate 500K TPD quota bucket, independent of the 70B model's quota |
| Widespread `TimeoutError`s during RAGAS evaluation even after fixing rate-limit-triggering bugs | Default RAGAS evaluation concurrency exceeded Groq's 30 requests/minute limit, particularly for metrics requiring multiple sequential LLM calls per sample (faithfulness, context recall) | Passed `RunConfig(max_workers=3, timeout=120)` to `evaluate()` to throttle concurrency |

---

## Known Limitations

- Single-domain corpus (Next.js documentation only); not tested against other documentation types or structures.
- No authentication on API endpoints — acceptable for a portfolio demo, not production-ready as-is.
- BM25 index requires an explicit `refresh_index()` call to pick up newly-ingested documents; this is not currently wired to happen automatically at the end of every `/ingest` call in all code paths.
- Token counting for chunk-size budgeting uses OpenAI's `tiktoken` encoding as an approximation, not Llama's actual tokenizer.
- A portion of RAGAS evaluation judge calls fail under free-tier rate limits even with throttled concurrency; aggregate scores are computed from the subset that succeeded.
- Render's free tier cold-starts after inactivity, adding 30–60 seconds of latency to the first request after idle.

## Future Work

- Langfuse observability integration (originally scoped as a stretch goal).
- Automatic BM25 index refresh on every successful ingest, without requiring a separate manual call.
- CI workflow to run the evaluation suite on a schedule or manual trigger (not yet wired as an actual GitHub Actions workflow file, given the dual-Python-version environment split between the main app and evaluation tooling).
- Support for additional documentation domains beyond Next.js, to validate the chunking and retrieval design generalizes.
- Native Qdrant hybrid search (sparse + dense vectors in a single collection) as a more architecturally elegant alternative to the current separate-then-merge BM25/vector approach.
