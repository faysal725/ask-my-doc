'use client';
import { useState } from 'react';
import { Upload, Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { ingestDocument } from '@/lib/api';

export default function UploadForm() {
  const [status, setStatus] = useState('idle'); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');

  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setStatus('loading');
    setResult(null);
    setErrorMsg('');

    try {
      const content = await file.text();
      const data = await ingestDocument(file.name, content);
      setResult(data);
      setStatus(data.errors?.length ? 'error' : 'success');
      if (data.errors?.length) setErrorMsg(data.errors.join(', '));
    } catch (err) {
      setStatus('error');
      setErrorMsg(err.message);
    }
  }

  return (
    <div className="border border-dashed border-[var(--border)] rounded-lg p-8 text-center">
      <input
        type="file"
        accept=".md,.mdx,.txt"
        id="file-upload"
        className="hidden"
        onChange={handleFile}
      />
      <label
        htmlFor="file-upload"
        className="flex flex-col items-center gap-3 cursor-pointer"
      >
        {status === 'loading' ? (
          <Loader2 size={32} className="animate-spin text-[var(--accent)]" />
        ) : status === 'success' ? (
          <CheckCircle2 size={32} className="text-green-500" />
        ) : status === 'error' ? (
          <XCircle size={32} className="text-red-500" />
        ) : (
          <Upload size={32} className="text-[var(--accent)]" />
        )}

        <p className="text-sm text-[var(--text)]">
          {status === 'loading' ? 'Ingesting document...' : 'Click to upload a .mdx or .md file'}
        </p>
        <p className="text-xs text-[var(--text-dim)] font-mono">supports markdown / mdx</p>
      </label>

      {result && status === 'success' && (
        <div className="mt-4 text-sm text-left bg-[var(--surface)] border border-[var(--border)] rounded-md p-3 font-mono">
          <p className="text-green-500">✓ {result.filename}</p>
          <p className="text-[var(--text-dim)] mt-1">
            {result.chunks} chunks · {result.embedded} embedded · {result.upserted} stored
          </p>
        </div>
      )}

      {status === 'error' && errorMsg && (
        <div className="mt-4 text-sm text-left bg-[var(--surface)] border border-red-900 rounded-md p-3 font-mono text-red-400">
          {errorMsg}
        </div>
      )}
    </div>
  );
}