'use client';
import { useState } from 'react';
import { ChevronDown, ChevronRight, FileText } from 'lucide-react';

export default function SourceCitation({ source, index }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border border-[var(--border)] rounded-md bg-[var(--surface)] overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left text-sm text-[var(--text-dim)] hover:text-[var(--text)] transition-colors"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <FileText size={14} className="text-[var(--accent)]" />
        <span className="font-mono text-xs">[{index + 1}]</span>
        <span className="truncate">{source.heading_path}</span>
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 border-t border-[var(--border)]">
          <p className="text-xs text-[var(--text-dim)] mb-2 font-mono">{source.source_doc}</p>
          <p className="text-sm text-[var(--text-dim)] whitespace-pre-wrap">{source.text}</p>
        </div>
      )}
    </div>
  );
}