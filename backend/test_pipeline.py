import asyncio
from app.services.chunking import chunk_markdown
from app.services.embedding import embed_documents
from app.services.vector_store import upsert_chunks


async def main():
    for filename in ["1.mdx", "2.mdx", "3.mdx"]:
        with open(f"data/{filename}", "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_markdown(content, filename)
        successful, failed = await embed_documents(chunks)
        total, errors = upsert_chunks(successful)
        print(f"{filename}: chunked={len(chunks)}, embedded={len(successful)}, upserted={total}, errors={errors}")


asyncio.run(main())