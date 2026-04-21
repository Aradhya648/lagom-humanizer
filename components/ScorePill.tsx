"use client";

interface ScorePillProps {
  score: number;
  label: string;
  size?: "sm" | "md";
}

function getColors(score: number) {
  if (score <= 30) {
    return {
      bg: "bg-emerald-950/60",
      border: "border-emerald-800/50",
      text: "text-emerald-400",
      dot: "bg-emerald-400",
    };
  }
  if (score <= 60) {
    return {
      bg: "bg-amber-950/60",
      border: "border-amber-800/50",
      text: "text-amber-400",
      dot: "bg-amber-400",
    };
  }
  if (score <= 80) {
    return {
      bg: "bg-orange-950/60",
      border: "border-orange-800/50",
      text: "text-orange-400",
      dot: "bg-orange-400",
    };
  }
  return {
    bg: "bg-red-950/60",
    border: "border-red-800/50",
    text: "text-red-400",
    dot: "bg-red-400",
  };
}

export default function ScorePill({ score, label, size = "md" }: ScorePillProps) {
  const colors = getColors(score);
  const isSmall = size === "sm";

  return (
    <span
      className={`score-pill inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-medium ${colors.bg} ${colors.border} ${colors.text} ${isSmall ? "text-[11px]" : "text-xs"}`}
    >
      <span className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${colors.dot}`} />
      <span className="tabular-nums">{score}</span>
      <span className="opacity-40">·</span>
      <span>{label}</span>
    </span>
  );
}
