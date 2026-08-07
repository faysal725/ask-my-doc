from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.core.config import settings
import uuid


client = QdrantClient(settings.qdrant_url, api_key=settings.qdrant_api_key)

COLLECTION_NAME = "ask_my_doc" 
BATCH_SIZE = 100

def ensure_collection() -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )


def _build_point(chunk: dict)-> PointStruct:
    # turn one chunk dict → PointStruct (id + vector + payload), ready for upsert.

    key= f"{chunk['source_doc']}::{chunk['chunk_index']}"
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    payload = {k: v for k, v in chunk.items() if k != "vector"}
    return PointStruct(
        id=point_id,
        vector=chunk["vector"],
        payload=payload
    )



def upsert_chunks(chunks: list[dict]) -> tuple[int, list[str]]:
    # build points from all chunks, batch, upsert each batch.
    

    ensure_collection()
    points = [_build_point(chunk) for chunk in chunks]

    total = 0
    errors: list[str] = []
    
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]

        try:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total += len(batch)
        except Exception as e:
            errors.append(f"batch {i}-{i+len(batch)} failed: {e}")

    return total, errors