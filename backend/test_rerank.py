import asyncio
from app.services.hybrid_retrieval import hybrid_search
from app.services.reranker import rerank


async def main():
    hybrid_results = await hybrid_search("how does prefetching work in next.js", final_k=15)
    reranked = rerank("how does prefetching work in next.js", hybrid_results, top_k=5)
    for r in reranked:
        print("---")
        print("rerank_score:", r["rerank_score"])
        print("heading:", r["heading_path"])


asyncio.run(main())