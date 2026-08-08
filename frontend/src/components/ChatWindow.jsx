'use client';
import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Loader2 } from 'lucide-react';
import { queryDocs } from '@/lib/api';
import SourceCitation from './SourceCitation';

export default function ChatWindow() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSend() {
    const query = input.trim();
    if (!query || loading) return;

    setMessages((prev) => [...prev, { role: 'user', content: query }]);
    setInput('');
    setLoading(true);

    try {
      const data = await queryDocs(query);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.answer, sources: data.sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}`, isError: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-[var(--text-dim)] mt-20">
            <p className="font-mono text-sm">$ ask-my-doc --query</p>
            <p className="mt-2 text-sm">Ask a question about the Next.js docs.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
            <div className={`max-w-2xl ${msg.role === 'user' ? 'bg-[var(--accent)] text-black rounded-lg px-4 py-2' : 'w-full'}`}>
              {msg.role === 'user' ? (
                <p className="text-sm">{msg.content}</p>
              ) : (
                <div className="space-y-3">
                  <div className={`prose prose-invert prose-sm max-w-none ${msg.isError ? 'text-red-400' : ''}`}>
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs text-[var(--text-dim)] font-mono">SOURCES</p>
                      {msg.sources.map((source, idx) => (
                        <SourceCitation key={idx} source={source} index={idx} />
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-[var(--text-dim)] text-sm">
            <Loader2 size={14} className="animate-spin" />
            <span className="font-mono">retrieving context...</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-[var(--border)] p-4">
        <div className="flex items-center gap-2 bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about revalidatePath, caching, routing..."
            className="flex-1 bg-transparent outline-none text-sm placeholder:text-[var(--text-dim)]"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="text-[var(--accent)] disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}