import asyncio
import random
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from app.core.config import settings


client = genai.Client(api_key=settings.gemini_api_key)

MODEL = "gemini-embedding-001"
DIMENSION = 768
MAX_CONCURRENT = 5
MAX_RETRIES = 5


try:
    client.models.embed_content(
        model="gemini-embedding-001",
        contents="test",
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768)
    )

except ClientError as e:
    print("type:", type(e))
    print("dir:", [a for a in dir(e) if not a.startswith("_")])
    print("str:", str(e))

async def _embed_single(text: str, task_type: str, chunk_id: str) -> dict:
    """Embed one text with retry+backoff. returns dict w/ chunk_id, vector, success"""
    last_error = None


    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.to_thread(
                client.models.embed_content,
                model=MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=DIMENSION
                )
            )

            return {
                "chunk_id": chunk_id,
                "vector": response.embeddings[0].values,
                "success": True
            }
        
        except ClientError as e:
            if e.code == 429:
                wait = min(2 ** attempt + random.uniform(0, 1), 30)
                await asyncio.sleep(wait)
                last_error = e 
                continue

            # 401, 400, etc - not retryable, fail now
            return {"chunk_id": chunk_id, "vector": None, "success": False, "error": str(e)}

        except ServerError as e:
            wait = min(2 ** attempt + random.uniform(0, 1), 30)
            await asyncio.sleep(wait)
            last_error = e
            continue 

        except Exception as e:
            wait = min(2 ** attempt + random.uniform(0, 1), 30)
            await asyncio.sleep(wait)
            last_error = e
            continue
    return {"chunk_id": chunk_id, "vector": None, "success": False, "error": f"max retries exceeded: {last_error}"}
        



async def _embed_batch_concurrent(items: list[tuple[str, int | str]], task_type: str) -> list[dict]:
    """run _embed_single over items, capped at MAX_CONCURRENT in flight."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def _bounded_embed(text, chunk_id):
        async with semaphore:
            return await _embed_single(text, task_type, chunk_id)

    tasks = [_bounded_embed(text, cid) for text, cid in items]
    results = await asyncio.gather(*tasks)

    return results


async def _embed_in_logical_batches(
    items: list[tuple[str, int | str]],
    task_type: str,
    batch_size: int = 50
) -> list[dict]:
    """Process items in logical batches of batch_size, concurrent within each."""
    all_results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await _embed_batch_concurrent(batch, task_type)
        all_results.extend(batch_results)

    return all_results




async def embed_documents(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Embed a list of chunk dicts (from your chunking function) for ingest.
    Returns (successful_chunks, failed_chunks).
    successful_chunks = original chunk dict + 'vector' key added.
    failed_chunks = original chunk dict + 'error' key added.    
    """

    items = [(c["text"], f"{c['source_doc']}::{c['chunk_index']}") for c in chunks]
    results = await _embed_in_logical_batches(items, task_type="RETRIEVAL_DOCUMENT")

    results_by_id = {r["chunk_id"]: r for r in results}

    successful, failed = [], []

    for chunk in chunks:
        key= f"{chunk['source_doc']}::{chunk['chunk_index']}"
        result = results_by_id[key]
        if result["success"]:
            successful.append({**chunk, "vector": result["vector"]})
        else:
            failed.append({**chunk, "error": result["error"]})

    return successful, failed


async def embed_query(text: str) -> list[float]:
    """Embed a single search query. async wrapper — search route is one request, no concurrency needed."""
    result = await _embed_single(text, task_type="RETRIEVAL_QUERY", chunk_id="query")
    if not result["success"]:
        raise RuntimeError(f"Query embedding failed: {result['error']}")
    return result["vector"]
