const API_URL = process.env.NEXT_PUBLIC_API_URL;

export async function ingestDocument(filename, content) {
  const res = await fetch(`${API_URL}/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, content }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Ingest failed: ${res.status}`);
  }
  return res.json();
}

export async function queryDocs(query, topK = 5) {
  const res = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Query failed: ${res.status}`);
  }
  return res.json();
}