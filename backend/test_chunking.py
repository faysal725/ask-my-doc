from app.services.chunking import chunk_markdown

with open("data/2.mdx", "r", encoding="utf-8") as f:
    content = f.read()

result = chunk_markdown(content, "2.mdx")
print(f"Total chunks: {len(result)}")
for c in result:
    print("---")
    print("chunk_index:", c["chunk_index"])
    print("heading_path:", c["heading_path"])
    print("contains_code:", c["contains_code"])
    print("tokens:", c["token_count"])