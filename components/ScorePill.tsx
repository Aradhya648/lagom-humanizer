"use client";

interface ScorePillProps {
  score: number;
  label: string;
  size?: "sm" | "md";
}

function getColors(score: number) {
  if (score <= 30) {
    return {
      bg: "bg-emerald-950/70",
      border: "border-emerald-700/40",
      text: "text-emerald-400",
      dot: "bg-emerald-400",
    };
  }
  if (score <= 60) {
    return {
      bg: "bg-amber-950/70",
      border: "border-amber-700/40",
      text: "text-amber-400",
      dot: "bg-amber-400",
    };
  }
  if (score <= 80) {
    return {
      bg: "bg-orange-950/70",
      border: "border-orange-700/40",
      text: "text-orange-400",
      dot: "bg-orange-400",
    };
  }
  return {
    bg: "bg-red-950/70",
    border: "border-red-800/40",
    text: "text-red-400",
    dot: "bg-red-400",
  };
}

export default function ScorePill({ score, label, size = "md" }: ScorePillProps) {
  const colors = getColors(score);

  return (
    <span
      className={`score-pill inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-medium ${colors.bg} ${colors.border} ${colors.text} ${size === "sm" ? "text-[11px]" : "text-xs"}`}
    >
      <span className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${colors.dot}`} />
      <span className="tabular-nums">{score}</span>
      <span className="opacity-40">·</span>
      <span>{label}</span>
    </span>
  );
}
