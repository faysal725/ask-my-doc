from pydantic import BaseModel


class IngestRequest(BaseModel):
    filename: str
    content: str


class IngestResponse(BaseModel):
    filename: str
    chunks: int
    embedded: int
    upserted: int
    errors: list[str] = []


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class SourceChunk(BaseModel):
    heading_path: str
    source_doc: str
    text: str
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]