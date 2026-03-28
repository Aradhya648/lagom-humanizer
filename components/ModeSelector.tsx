"use client";

export type Mode = "light" | "medium" | "aggressive";

interface ModeSelectorProps {
  value: Mode;
  onChange: (mode: Mode) => void;
}

const modes: { id: Mode; label: string; tooltip: string }[] = [
  {
    id: "light",
    label: "Light",
    tooltip: "Gentle edits — smooths out stiff phrasing while keeping your original structure and tone.",
  },
  {
    id: "medium",
    label: "Medium",
    tooltip: "A real rewrite — different rhythm, fresher word choices, still your argument.",
  },
  {
    id: "aggressive",
    label: "Aggressive",
    tooltip: "Full voice overhaul. It'll read like you actually sat down and wrote the thing.",
  },
];

export default function ModeSelector({ value, onChange }: ModeSelectorProps) {
  return (
    <div className="flex items-center gap-1 p-1 rounded-xl bg-surface border border-border">
      {modes.map((mode) => (
        <div key={mode.id} className="relative group flex-1">
          <button
            onClick={() => onChange(mode.id)}
            className={`w-full px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
              value === mode.id
                ? "bg-accent text-background"
                : "text-muted hover:text-text"
            }`}
          >
            {mode.label}
          </button>
          {/* Tooltip */}
          <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-52 z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <div className="bg-[#1a1a1a] border border-border rounded-lg px-3 py-2 text-xs text-muted leading-relaxed shadow-xl">
              {mode.tooltip}
            </div>
            <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-border" />
          </div>
        </div>
      ))}
    </div>
  );
}
