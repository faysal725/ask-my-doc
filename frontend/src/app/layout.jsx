import './globals.css';

export const metadata = {
  title: 'Ask My Doc',
  description: 'RAG-powered Q&A for Next.js documentation',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}