import Link from 'next/link';
import { Upload } from 'lucide-react';
import ChatWindow from '@/components/ChatWindow';

export default function Home() {
  return (
    <main className="h-screen flex flex-col">
      <header className="border-b border-[var(--border)] px-4 py-3 flex items-center justify-between">
        <h1 className="font-mono text-sm font-semibold">
          ask<span className="text-[var(--accent)]">my</span>doc
        </h1>
        <Link
          href="/upload"
          className="flex items-center gap-1.5 text-xs text-[var(--text-dim)] hover:text-[var(--text)] border border-[var(--border)] rounded-md px-3 py-1.5"
        >
          <Upload size={12} />
          Upload docs
        </Link>
      </header>
      <div className="flex-1 overflow-hidden">
        <ChatWindow />
      </div>
    </main>
  );
}