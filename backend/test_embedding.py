import asyncio
from app.services.embedding import embed_query, embed_documents


async def main():
    # test embed_query
    query_vector = await embed_query("How does caching work in Next.js?")
    print("Query embedding dim:", len(query_vector))

    # test embed_documents with fake chunk dicts
    fake_chunks = [
        {"text": "Next.js caches data automatically.", "source_doc": "test.mdx", "chunk_index": 0},
        {"text": "Use revalidatePath to invalidate cache.", "source_doc": "test.mdx", "chunk_index": 1},
    ]
    successful, failed = await embed_documents(fake_chunks)
    print(f"Successful: {len(successful)}, Failed: {len(failed)}")
    if successful:
        print("First chunk vector dim:", len(successful[0]["vector"]))


asyncio.run(main())