import cohere
from app.core.config import settings

client = cohere.ClientV2(api_key=settings.cohere_api_key)

MODEL = "rerank-v3.5"


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []

    documents = [chunk["text"] for chunk in chunks]

    response = client.rerank(
        model=MODEL,
        query=query,
        documents=documents,
        top_n=top_k,
    )

    results = []
    for item in response.results:
        chunk = chunks[item.index]
        results.append({**chunk, "rerank_score": item.relevance_score})

    return results