import tiktoken
import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")

# Using tiktoken (OpenAI's tokenizer) as an approximation for chunk sizing.
# Llama's actual tokenizer differs slightly, but token counts are close
# enough for chunk-size budgeting purposes.
# Initialize the tiktoken encoder for the "cl100k_base" encoding.
_encoder = tiktoken.get_encoding("cl100k_base")

FENCE_RE = re.compile(r"^```")


# Count the number of tokens in a string using the tiktoken library.
def count_tokens(text:str) -> int:
    return len(_encoder.encode(text))


def _build_heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(text for _, text in stack)



def _split_oversized(chunk_dict: dict, max_tokens: int = 500, start_index: int = 0) -> list[dict]:
    if chunk_dict["token_count"] <= max_tokens:
        return [chunk_dict]

    paragraphs = chunk_dict["text"].split("\n\n")

    # merge paragraphs back together if a fence spans across a \n\n boundary
    safe_paragraphs: list[str] = []
    buf = ""
    fence_open = False

    for para in paragraphs:
        fence_count = para.count("```")
        buf = para if not buf else buf + "\n\n" + para

        if fence_count % 2 == 1:
            fence_open = not fence_open

        if not fence_open:
            safe_paragraphs.append(buf)
            buf = ""

    if buf:
        safe_paragraphs.append(buf)

    sub_chunks: list[dict] = []
    current: list[str] = []
    current_tokens = 0
    sub_index = 0

    for para in safe_paragraphs:
        para_tokens = count_tokens(para)

        if current and current_tokens + para_tokens > max_tokens:
            sub_text = "\n\n".join(current).strip()
            sub_chunks.append({
                "text": sub_text,
                "heading_path": chunk_dict["heading_path"],
                "token_count": count_tokens(sub_text),
                "contains_code": "```" in sub_text,
                "chunk_index": start_index + sub_index,
                "source_doc": chunk_dict["source_doc"],
            })
            sub_index += 1
            current = []
            current_tokens = 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        sub_text = "\n\n".join(current).strip()
        sub_chunks.append({
            "text": sub_text,
            "heading_path": chunk_dict["heading_path"],
            "token_count": count_tokens(sub_text),
            "contains_code": "```" in sub_text,
            "chunk_index": start_index + sub_index,
            "source_doc": chunk_dict["source_doc"],
        })

    return sub_chunks


# Chunk a markdown text into smaller pieces based on token count and line breaks.
def chunk_markdown(text: str, source_doc: str) -> list[dict]:
    # ... frontmatter strip from before ...
    lines = text.split("\n")
    doc_title = source_doc  # fallback default

    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                closing_idx = i
                break
        else:
            closing_idx = None

        if closing_idx is not None:
            frontmatter = lines[1:closing_idx]
            for fm_line in frontmatter:
                if fm_line.strip().startswith("title:"):
                    doc_title = fm_line.split(":", 1)[1].strip().strip('"\'')
                    break
            lines = lines[closing_idx + 1:]

    heading_stack: list[tuple[int, str]] = []
    if doc_title != source_doc:  # only seed if real title found, not fallback
        heading_stack.append((1, doc_title))

    current_buffer: list[str] = []
    chunk_index = 0
    chunks: list[dict] = []
    in_code_fence = False
    in_table = False
    has_code = False

    for line in lines:
        stripped = line.strip()

        # --- fence toggle ---
        if FENCE_RE.match(stripped):
            in_code_fence = not in_code_fence
            current_buffer.append(line)
            has_code = True
            continue

        # if inside a fence, skip ALL other checks — just buffer the line
        if in_code_fence:
            current_buffer.append(line)
            continue

        # table toggle (only reached when NOT in a fence)
        if stripped.startswith("|"):
            current_buffer.append(line)
            continue
        else:
            in_table = False

        if in_table:
            current_buffer.append(line)
            continue

        match = HEADING_RE.match(line)

        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            if current_buffer:
                chunk_text = "\n".join(current_buffer).strip()
                if chunk_text:
                    new_chunk = {
                        "text": chunk_text,
                        "heading_path": _build_heading_path(heading_stack),
                        "token_count": count_tokens(chunk_text),
                        "contains_code": has_code,
                        "chunk_index": chunk_index,
                        "source_doc": source_doc,
                    }
                    split_result = _split_oversized(new_chunk, start_index=chunk_index)
                    chunks.extend(split_result)
                    chunk_index += len(split_result)
                current_buffer = []
                has_code = False

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))

        else:
            current_buffer.append(line)

    if current_buffer:
        chunk_text = "\n".join(current_buffer).strip()
        if chunk_text:
            new_chunk = {
                "text": chunk_text,
                "heading_path": _build_heading_path(heading_stack),
                "token_count": count_tokens(chunk_text),
                "contains_code": has_code,
                "chunk_index": chunk_index,
                "source_doc": source_doc,
            }
            split_result = _split_oversized(new_chunk, start_index=chunk_index)
            chunks.extend(split_result)

    return chunks