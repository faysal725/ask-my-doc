
from app.services.bm25_index import search_bm25, get_or_build_index
from app.services.vector_store import search_vectors
from app.services.embedding import embed_query



async def hybrid_search(query: str, top_k_dense=30, top_k_sparse=30, final_k=10, rrf_k=60) -> list[dict]:
    # perform hybrid search: dense + sparse, return final top-k results.

    query_vector = await embed_query(query)
    dense_results = search_vectors(query_vector, top_k_dense)

    bm25_index, bm25_chunks = get_or_build_index()
    sparse_results = search_bm25(query, bm25_index, bm25_chunks, top_k_sparse)

    vector_ranks = {result["chunk_id"]: rank for rank, result in enumerate(dense_results)}
    bm25_ranks = {result["chunk_id"]: rank for rank, result in enumerate(sparse_results)} 
