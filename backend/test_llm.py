import asyncio
from app.services.hybrid_retrieval import hybrid_search
from app.services.reranker import rerank
from app.services.llm_client import generate_answer


async def main():
    query = "how does prefetching work in next.js"
    hybrid_results = await hybrid_search(query, final_k=15)
    reranked = rerank(query, hybrid_results, top_k=5)
    answer = generate_answer(query, reranked)
    print(answer)


asyncio.run(main())