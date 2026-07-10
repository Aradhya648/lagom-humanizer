import type { Metadata, Viewport } from "next";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export const metadata: Metadata = {
  title: "lagom: Just the right amount of human",
  description:
    "Transform AI-generated text into natural, human-sounding writing. Academic and general writing modes.",
  keywords: ["AI humanizer", "text humanizer", "academic writing", "AI detection", "humanize AI text"],
  openGraph: {
    title: "lagom : Just the right amount of human",
    description: "Transform AI-generated text into natural, human-sounding writing.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Lora:ital@0;1&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-background text-text font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
