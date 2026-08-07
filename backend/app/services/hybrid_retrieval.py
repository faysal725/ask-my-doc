from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.services.bm25_index import search_bm25
from app.core.config import settings
from app.services.embedding import embed_query


client = QdrantClient(settings.qdrant_url, api_key=settings.qdrant_api_key)


def search_vectors(query_vector: list[float], top_k: int = 10) -> list[dict]:
    if not query_vector:
        raise ValueError("query_vector empty")
    try:
        response = client.query_points(
            collection_name="nextjs_docs_v1",
            query=query_vector,
            limit=top_k,
            with_payload=True
            )
    except Exception as e:
        # TODO: split httpx transport errors vs Qdrant API errors, like your upsert fn does
        # TODO: log error
        return []

    results = []
    for point in response.points:
        results.append({
            "chunk_id": point.payload["chunk_id"],
            "score": point.score,
            "payload": point.payload
        })
    return results


async def hybrid_search(query: str, top_k_dense=30, top_k_sparse=30, final_k=10, rrf_k=60)-> list[dict]:
    # perform hybrid search: dense + sparse, return final top-k results.

    query_vector = await embed_query(query)
    dense_results = search_vectors(query_vector, top_k_dense)
    sparse_results = search_bm25(query, top_k_sparse)

    vector_ranks = {result["chunk_id"]: rank for rank, result in enumerate(dense_results)}
    bm25_ranks = {result["chunk_id"]: rank for rank, result in enumerate(sparse_results)}

    
