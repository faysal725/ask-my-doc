
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
    all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

    chunk_data_lookup = {}
    for result in dense_results:
        chunk_data_lookup[result["chunk_id"]] = result["payload"]
    for result in sparse_results:
        chunk_data_lookup[result["chunk_id"]] = result

    rrf_scores = []
    for chunk_id in all_chunk_ids:
        score = 0.0
        if chunk_id in vector_ranks:
            score += 1 / (rrf_k + vector_ranks[chunk_id] + 1)
        if chunk_id in bm25_ranks:
            score += 1 / (rrf_k + bm25_ranks[chunk_id] + 1)
        rrf_scores.append((chunk_id, score))

    rrf_scores.sort(key=lambda pair: pair[1], reverse=True)
    top_results = rrf_scores[:final_k]

    final_results = []
    for chunk_id, score in top_results:
        chunk_data = chunk_data_lookup[chunk_id]
        final_results.append({**chunk_data, "rrf_score": score, "chunk_id": chunk_id})

    return final_results
