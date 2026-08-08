# Ask My Doc — Production RAG System

A hybrid-retrieval RAG assistant for Next.js documentation, built with vector + keyword search, cross-encoder reranking, and RAGAS-evaluated groundedness — deployed end-to-end on free-tier infrastructure.

## Demo

Live: [https://ask-my-doc-ten.vercel.app](https://ask-my-doc-ten.vercel.app)

## Problem & Motivation

Built while transitioning from frontend development (React/Next.js) into AI engineering. The goal: a documentation Q&A assistant that gives grounded, cited answers instead of hallucinated ones — and a system realistic enough to demonstrate production RAG engineering decisions, not just a tutorial-level demo.

Next.js documentation was chosen deliberately as the domain: it mixes prose, code blocks, and tables (a real stress test for chunking), contains exact-match technical terms like function and config names (justifying hybrid retrieval over vector-only search), and — since it's the author's own area of expertise — allows for a trustworthy, hand-written evaluation set rather than guessed-at ground truth.

## Architecture


**Ingestion pipeline**
- Documents are split using a custom heading-aware markdown chunker that never breaks a code fence or table mid-block, and attaches a full heading path (e.g. `Prefetching > Controlling prefetching > Manual prefetch`) to every chunk as metadata.
- Chunks are embedded via Gemini's `gemini-embedding-001` model and stored in Qdrant Cloud with deterministic UUIDs, making re-ingestion idempotent (safe to re-run without creating duplicates).

**Retrieval pipeline**
- Dense (vector similarity) and sparse (BM25 keyword) search run independently, then are merged using **Reciprocal Rank Fusion (RRF)** — combining rank positions rather than raw scores, since cosine similarity and BM25 scores live on incompatible scales.
- The fused candidate set is passed through a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which scores the query and each chunk jointly — catching relevance signals that bi-encoder embedding similarity alone misses.
- The top reranked chunks are passed to Groq (Llama 3.3 70B) with a system prompt constraining the model to answer only from provided context and cite sources by heading path.

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Backend | FastAPI | Async-first, clean layered architecture, auto-generated OpenAPI docs |
| Vector DB | Qdrant Cloud (free tier) | Production-grade vector search, generous free tier, native hybrid-search support for future use |
| Embeddings | Gemini `gemini-embedding-001` | Free tier via AI Studio, no card required |
| Keyword search | `rank-bm25` | Lightweight, in-memory, no external service needed |
| Reranking | `sentence-transformers` cross-encoder | Runs on CPU, no GPU required |
| LLM generation | Groq (Llama 3.3 70B) | Free tier, extremely fast inference |
| Evaluation | RAGAS | LLM-judged faithfulness/relevancy/recall/precision, reference-free where possible |
| Frontend | Next.js / React | Chat + upload UI, deployed on Vercel free tier |

## Key Engineering Decisions

- **No chunk overlap.** Since chunking is heading-aware rather than naive fixed-size, section boundaries already preserve context — overlap (typically used to compensate for arbitrary cut points) was judged unnecessary here, trading a small amount of potential context loss for simpler, more predictable chunk boundaries.
- **Composite chunk IDs (`source_doc::chunk_index`) and deterministic UUIDs.** Prevents ID collisions across multi-document ingestion and makes re-ingesting the same document idempotent — critical for iterative development, where re-running ingest is common.
- **RRF over raw score combination.** Cosine similarity (~0–1 range) and BM25 scores (unbounded) cannot be meaningfully averaged. RRF sidesteps this by fusing rank positions instead of scores, and naturally handles chunks that appear in only one retrieval method's results without special-case logic.
- **Two-stage retrieval (broad hybrid search → precise reranking).** Bi-encoder embeddings compare query and chunk independently — fast but imprecise. A cross-encoder scores them jointly — slower, but far more accurate. Running the expensive cross-encoder only on the ~15-candidate hybrid shortlist (not the full corpus) balances speed and precision.
- **Isolated Python 3.12 environment for evaluation.** The main backend runs on Python 3.14; RAGAS's dependency tree (`scikit-network`, `dill`) had unresolved compatibility issues on 3.14 at time of writing. Rather than downgrading the whole backend, the eval suite runs in its own isolated virtual environment — a pragmatic tradeoff that keeps the production app on the newer Python version without blocking on an upstream library gap.

## Evaluation Results

Evaluated against a hand-written set of 20 question/answer pairs covering the ingested documentation (prefetching, ISR/Cache Components, Babel configuration), using RAGAS with an LLM-judge (Groq Llama 3.1 8B).

| Metric | Score |
|---|---|
| Faithfulness | 0.88 |
| Answer Relevancy | 0.82 |
| Context Recall | 0.96 |
| Context Precision | 0.88 |

**Methodology notes:** Faithfulness measures whether generated claims are actually supported by retrieved context (catches hallucination); Context Recall measures whether retrieval found all information needed to answer correctly; Answer Relevancy measures whether the answer actually addresses the question asked. High context recall (0.96) combined with strong faithfulness (0.88) indicates the hybrid retrieval + reranking pipeline is reliably surfacing the right information, and generation stays grounded in it.

A portion of evaluation judge calls failed under free-tier rate limits (30 requests/minute on the evaluator model) and were excluded from the aggregate — an honest constraint of running evaluation entirely on free-tier infrastructure, not a hidden pipeline issue.

## API Reference

**`POST /ingest`**
```json
{ "filename": "example.mdx", "content": "..." }
```
Returns chunk/embed/store counts and any errors.

**`POST /query`**
```json
{ "query": "how does prefetching work?", "top_k": 5 }
```
Returns a grounded answer with cited source chunks.

## Local Setup

**Backend**
```bash
cd backend
poetry install
cp .env.example .env   # fill in GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY
poetry run uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL
npm run dev
```

**Evaluation** (requires a separate Python 3.12 environment — see `eval/README.md`)
```bash
cd eval
python -m venv .venv
source .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install ragas langchain-openai langchain-google-genai groq qdrant-client python-dotenv pydantic-settings sentence-transformers google-genai tiktoken rank-bm25
python run_ragas_eval.py
```

## Limitations & Future Work

- Single-domain corpus (Next.js docs only); not yet tested against other documentation types.
- No authentication on API endpoints.
- Langfuse observability (stretch goal) not yet implemented.
- BM25 index rebuilds only on-demand (`refresh_index()`), not automatically on every ingest without explicit call — acceptable for current scale, would need a more robust cache-invalidation strategy at larger scale.
- CI currently runs the eval suite on manual trigger rather than every commit, due to the dual-Python-version environment split between the main app and the eval tooling.

## License

MIT