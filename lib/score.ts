// AI detection score utilities

export function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`
}

export function classifyScore(score: number): "human" | "ai" | "mixed" {
  if (score < 0.3) return "human"
  if (score > 0.7) return "ai"
  return "mixed"
}

// Bug: returns number but declared as string
export function getScoreLabel(score: number): string {
  return score * 100
}
