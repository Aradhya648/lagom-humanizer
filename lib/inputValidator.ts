// ── FUTURE CHUNKING HOOK ─────────────────────────────────────────────
// When ready to support >1000 words:
//   1. Add chunkText(text, chunkSize): string[] here
//   2. In pipeline.ts, map each chunk through runRewritePipeline
//   3. Join chunk outputs with a final coherence pass
// This file is the single place that controls the input boundary.
// ─────────────────────────────────────────────────────────────────────

import { countWords, WORD_LIMIT } from './wordCount'

export type ValidationResult =
  | { valid: true }
  | { valid: false; error: string }

export function validateInput(text: string): ValidationResult {
  const trimmed = text.trim()

  if (!trimmed) {
    return { valid: false, error: 'Please enter some text to humanize.' }
  }

  if (trimmed.length < 20) {
    return {
      valid: false,
      error: 'Text is too short — please enter at least a sentence or two.',
    }
  }

  const words = countWords(trimmed)
  if (words > WORD_LIMIT) {
    return {
      valid: false,
      error: `Your text is ${words} words. The current limit is ${WORD_LIMIT} words. Please shorten your input.`,
    }
  }

  return { valid: true }
}
