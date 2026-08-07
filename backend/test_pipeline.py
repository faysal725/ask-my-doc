from app.services.chunking import chunk_markdown
from app.services.bm25_index import build_bm25_index, search_bm25

with open("data/3.mdx", "r", encoding="utf-8") as f:
    content = f.read()

chunks = chunk_markdown(content, "3.mdx")
index = build_bm25_index(chunks)

results = search_bm25("prefetch", index, chunks, top_k=3)
for r in results:
    print("---")
    print("score:", r["bm25_score"])
    print("heading:", r["heading_path"])
    print(r["text"][:150])