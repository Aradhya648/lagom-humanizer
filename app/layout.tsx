import type { Metadata } from 'next'
import '@/styles/globals.css'

export const metadata: Metadata = {
  title: 'Lagom — Write just right',
  description: 'Transform AI-generated text into natural, human-quality academic writing. Just the right amount of rewriting.',
  keywords: ['AI humanizer', 'academic writing', 'text rewriter', 'student writing'],
  openGraph: {
    title: 'Lagom — Write just right',
    description: 'Transform AI-generated text into natural, human-quality academic writing.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-lagom-bg text-lagom-ink antialiased">
        {children}
      </body>
    </html>
  )
}
