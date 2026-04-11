"use client";

import { type ContentType } from "@/prompts/pipeline";

export type { ContentType };

interface ContentTypeSelectorProps {
  value: ContentType;
  onChange: (contentType: ContentType) => void;
}

const CONTENT_TYPES: { id: ContentType; label: string; tooltip: string }[] = [
  {
    id: "general",
    label: "General Writing",
    tooltip: "Maximum variation — no constraints on rhythm or register. Best for blog posts, articles, and everyday writing.",
  },
  {
    id: "essay",
    label: "Essay",
    tooltip: "Academic essay — preserves complex clause structure and formal register. No contractions.",
  },
  {
    id: "academic",
    label: "Academic Paper",
    tooltip: "Research writing — calibrated for papers and reports. Keeps technical vocabulary and formal connectors.",
  },
  {
    id: "email",
    label: "Email",
    tooltip: "Conversational rhythm — natural contractions, direct tone. Reads like a real person writing to a colleague.",
  },
  {
    id: "document",
    label: "Document",
    tooltip: "Formal document — professional tone throughout. No contractions, precise vocabulary, clear structure.",
  },
];

export default function ContentTypeSelector({ value, onChange }: ContentTypeSelectorProps) {
  return (
    <div className="flex items-center gap-1 p-1 rounded-xl bg-surface border border-border">
      {CONTENT_TYPES.map((ct) => (
        <div key={ct.id} className="relative group flex-1">
          <button
            onClick={() => onChange(ct.id)}
            className={`w-full px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
              value === ct.id
                ? "bg-accent text-background"
                : "text-muted hover:text-text"
            }`}
          >
            {ct.label}
          </button>
          {/* Tooltip */}
          <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-52 z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <div className="bg-[#1a1a1a] border border-border rounded-lg px-3 py-2 text-xs text-muted leading-relaxed shadow-xl">
              {ct.tooltip}
            </div>
            <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-border" />
          </div>
        </div>
      ))}
    </div>
  );
}
