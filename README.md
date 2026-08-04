**# Ask My Doc — Production RAG System**

**One-line pitch**: A hybrid-retrieval RAG assistant for Next.js/React documentation, built with vector + keyword search, cross-encoder reranking, and RAGAS-evaluated groundedness — end-to-end on free-tier infra.

**## Demo**
- Live link (Vercel frontend)
- Screenshot/GIF of chat UI with cited answer

**## Problem & Motivation**
- Short paragraph: why this exists (your real transition story — navigating framework docs while learning AI eng)
- What "good" means here (grounded answers, cited sources, refuses when context insufficient)

**## Architecture**
- Diagram (draw.io/excalidraw export or ASCII) showing: Upload → Chunk → Embed → Qdrant | Query → Hybrid Retrieval (BM25+vector) → Rerank → LLM (Groq) → Cited Answer
- Short explanation per stage, WHY each design choice (heading-aware chunking, hybrid retrieval, rerank) — this section is what interviewers actually read

**## Tech Stack**
- Table: component → tool → why chosen (e.g. Qdrant Cloud — free tier, production-grade vector DB vs. toy in-memory store)

**## Key Engineering Decisions**
- 3-5 bullets on nontrivial tradeoffs: chunking strategy, embedding provider choice, why hybrid over vector-only, rerank model choice, free-tier constraint handling (rate limits, caching)
- This is the "senior thinking" section — most important for CV signal

**## Evaluation Results**
- RAGAS metrics table: faithfulness, answer relevancy, context precision/recall — baseline numbers
- Short note on eval methodology (20-50 QA pairs, how generated/validated)
- CI badge/link showing eval runs automatically

**## API Reference**
- `POST /ingest`, `POST /query` — request/response shape, brief

**## Local Setup**
- Clone, env vars needed (`.env.example` reference), backend run cmd, frontend run cmd

**## Limitations & Future Work**
- Honest section: known gaps (e.g. no auth, single-doc-type corpus, Langfuse tracing as stretch not yet done)
- Signals maturity — shows you know it's not "done," not overclaiming

**## License**