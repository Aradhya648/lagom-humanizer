export const WORD_LIMIT = 1000

export function countWords(text: string): number {
  const trimmed = text.trim()
  if (!trimmed) return 0
  return trimmed.split(/\s+/).length
}

export function isWithinLimit(text: string): boolean {
  return countWords(text) <= WORD_LIMIT
}
