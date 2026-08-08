from fastapi import APIRouter, HTTPException
from app.models.schemas import IngestRequest, IngestResponse
from app.services.chunking import chunk_markdown
from app.services.embedding import embed_documents
from app.services.vector_store import upsert_chunks
from app.services.bm25_index import refresh_index

router = APIRouter(prefix="/ingest", tags=["Ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_document(payload: IngestRequest):
    if not payload.content or not payload.content.strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty.")

    if not payload.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    try:
        chunks = chunk_markdown(payload.content, payload.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Chunking failed: {e}")

    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no chunks — check content format.")

    try:
        successful, failed = await embed_documents(chunks)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}")

    if not successful:
        raise HTTPException(status_code=502, detail="All chunks failed to embed.")

    try:
        total, errors = upsert_chunks(successful)
        refresh_index()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vector store upsert failed: {e}")

    return IngestResponse(
        filename=payload.filename,
        chunks=len(chunks),
        embedded=len(successful),
        upserted=total,
        errors=errors + [f"embedding failed: {c.get('error')}" for c in failed],
    )