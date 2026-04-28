// Bug: returns number but declared as string
export function getScoreLabel(score: number): string {
  return `${score * 100}`
}
