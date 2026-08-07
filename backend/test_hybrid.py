import asyncio
from app.services.hybrid_retrieval import hybrid_search


async def main():
    results = await hybrid_search("how does prefetching work in next.js")
    for r in results:
        print("---")
        print("rrf_score:", r["rrf_score"])
        print("heading:", r["heading_path"])
        print(r["text"][:150])


asyncio.run(main())