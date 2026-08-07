from sentence_transformers import CrossEncoder

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []

    model = _get_model()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    paired = list(zip(chunks, scores))
    paired.sort(key=lambda pair: pair[1], reverse=True)

    top_results = paired[:top_k]
    return [{**chunk, "rerank_score": float(score)} for chunk, score in top_results]