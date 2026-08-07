from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError
import time
from app.core.config import settings

client = Groq(api_key=settings.groq_api_key)

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a documentation assistant. Answer ONLY using the provided context chunks.
If the context doesn't contain enough information to answer, say so clearly — do not make up information.
Always cite which source chunk(s) you used by referencing their heading_path.
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[Source {i+1}: {chunk['heading_path']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have enough context to answer that question."

    context = build_context(chunks)

    user_prompt = f"""Context:
{context}

Question: {query}

Answer the question using only the context above. Cite sources by their heading_path."""

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            wait = 2 ** attempt
            time.sleep(wait)
            last_error = e

        except APIConnectionError as e:
            wait = 2 ** attempt
            time.sleep(wait)
            last_error = e

        except APIError as e:
            # non-retryable (bad request, auth, etc.)
            raise RuntimeError(f"Groq API error: {e}")

    raise RuntimeError(f"LLM generation failed after {MAX_RETRIES} retries: {last_error}")