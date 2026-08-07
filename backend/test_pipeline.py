import asyncio
from app.services.chunking import chunk_markdown
from app.services.embedding import embed_documents
from app.services.vector_store import upsert_chunks


async def main():
    with open("data/2.mdx", "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_markdown(content, "2.mdx")
    print(f"Chunked: {len(chunks)} chunks")

    successful, failed = await embed_documents(chunks)
    print(f"Embedded: {len(successful)} success, {len(failed)} failed")

    total, errors = upsert_chunks(successful)
    print(f"Upserted: {total} points, {len(errors)} batch errors")
    if errors:
        print(errors)


asyncio.run(main())