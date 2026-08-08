from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.hybrid_retrieval import hybrid_search
from app.services.reranker import rerank
from app.services.llm_client import generate_answer

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_docs(payload: QueryRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        hybrid_results = await hybrid_search(payload.query, final_k=15)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e}")

    if not hybrid_results:
        return QueryResponse(
            answer="I couldn't find any relevant information to answer that question.",
            sources=[],
        )

    try:
        reranked = rerank(payload.query, hybrid_results, top_k=payload.top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Reranking failed: {e}")

    try:
        answer = generate_answer(payload.query, reranked)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    sources = [
        SourceChunk(
            heading_path=c["heading_path"],
            source_doc=c["source_doc"],
            text=c["text"],
            rerank_score=c.get("rerank_score"),
        )
        for c in reranked
    ]

    return QueryResponse(answer=answer, sources=sources)