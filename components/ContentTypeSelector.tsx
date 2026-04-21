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
    label: "General",
    tooltip: "Maximum variation — no constraints on rhythm or register. Best for blog posts, articles, and everyday writing.",
  },
  {
    id: "essay",
    label: "Essay",
    tooltip: "Academic essay — preserves complex clause structure and formal register. No contractions.",
  },
  {
    id: "academic",
    label: "Academic",
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
    <div className="inline-flex items-center gap-0.5 p-1 rounded-xl bg-background border border-border shadow-panel">
      {CONTENT_TYPES.map((ct) => (
        <div key={ct.id} className="relative group">
          <button
            onClick={() => onChange(ct.id)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 whitespace-nowrap ${
              value === ct.id
                ? "bg-surface-raised text-text shadow-sm"
                : "text-muted hover:text-text hover:bg-surface"
            }`}
          >
            {ct.label}
          </button>
          {/* Tooltip */}
          <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2.5 w-56 z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <div className="bg-background border border-border rounded-xl px-3.5 py-2.5 text-xs text-muted leading-relaxed shadow-panel">
              {ct.tooltip}
            </div>
            <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-[5px] border-r-[5px] border-t-[5px] border-l-transparent border-r-transparent border-t-border" />
          </div>
        </div>
      ))}
    </div>
  );
}
