from rank_bm25 import BM25Okapi
import re



def _tokenize(text: str) -> list[str]:
    lower_case_word = text.lower()
    return re.findall(r"\w+", lower_case_word)



def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    # build BM25 index from chunks, return the index object.

    tokenized_document_list = [ _tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_document_list)



def search_bm25(query: str, index: BM25Okapi, chunks: list[dict], top_k: int = 10) -> list[dict]:
    tokenize_query = _tokenize(query)
    scores = index.get_scores(tokenize_query)
    paired = list(zip(chunks, scores))
    paired.sort(key=lambda pair: pair[1], reverse=True)

    top_results = paired[:top_k]
    results = []
    for chunk, score in top_results:
        result_chunk = {**chunk, "bm25_score": score}
        results.append(result_chunk)

    return results