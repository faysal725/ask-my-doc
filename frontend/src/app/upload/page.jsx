import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import UploadForm from '@/components/UploadForm';

export default function UploadPage() {
  return (
    <main className="min-h-screen max-w-2xl mx-auto px-4 py-10">
      <Link href="/" className="inline-flex items-center gap-2 text-sm text-[var(--text-dim)] hover:text-[var(--text)] mb-8">
        <ArrowLeft size={16} />
        Back to chat
      </Link>
      <h1 className="text-2xl font-mono font-semibold mb-2">Add documentation</h1>
      <p className="text-[var(--text-dim)] text-sm mb-8">
        Upload a Next.js docs page to make it searchable.
      </p>
      <UploadForm />
    </main>
  );
}