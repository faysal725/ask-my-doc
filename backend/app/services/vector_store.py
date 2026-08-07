from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.core.config import settings
import uuid
import time


client = QdrantClient(settings.qdrant_url, api_key=settings.qdrant_api_key)

COLLECTION_NAME = "nextjs_docs_v1" 
BATCH_SIZE = 100

def ensure_collection() -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )


def _build_point(chunk: dict)-> PointStruct:
    # turn one chunk dict → PointStruct (id + vector + payload), ready for upsert.

    
    if not chunk.get("vector"):
        raise ValueError(f"Chunk missing vector, cannot upsert: {chunk.get('source_doc')}::{chunk.get('chunk_index')}")

    key= f"{chunk['source_doc']}::{chunk['chunk_index']}"
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    payload = {k: v for k, v in chunk.items() if k != "vector"}
    payload["chunk_id"] = key   # ← new line
    return PointStruct(
        id=point_id,
        vector=chunk["vector"],
        payload=payload
    )

def fetch_all_chunks() -> list[dict]:
    all_chunks = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_chunks.extend(point.payload for point in points)
        if offset is None:
            break

    return all_chunks


def upsert_chunks(chunks: list[dict]) -> tuple[int, list[str]]:
    # build points from all chunks, batch, upsert each batch.
    

    ensure_collection()
    points = [_build_point(chunk) for chunk in chunks]

    total = 0
    errors: list[str] = []
    
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        last_error = None


        for attempt in range(2):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=batch)
                total += len(batch)
                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(1)  # wait a second before retrying

        if last_error:
            errors.append(f"batch {i}-{i+len(batch)} failed after retries: {last_error}")

    return total, errors