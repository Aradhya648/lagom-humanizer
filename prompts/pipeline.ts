export type ContentType = "essay" | "academic" | "email" | "document" | "general";

// Kept for backward compatibility — currently imported by lib/humanizer.ts.
export type SourceRegister = "academic" | "formal" | "neutral" | "informal";

// ─── Chunk Variation Hints ────────────────────────────────────────────────────

const CHUNK_HINTS = [
  "Open with a short direct sentence under 10 words.",
  "Let one sentence in this section run long — over 28 words.",
  "Keep this section punchy. Shorter sentences throughout.",
  "Vary the first word of every sentence — no two the same.",
  "Make the last sentence of this section shorter than the rest.",
];

// ─── Pass 1+2 Combined: Structural + Semantic in one call ─────────────────────
// Merging both passes halves LLM calls, critical for Vercel 10s timeout.

export function getCombinedPrompt(
  text: string,
  contentType: ContentType,
  chunkIndex: number,
  register?: SourceRegister
): string {
  const hint = CHUNK_HINTS[chunkIndex % 5];
  const wordCount = text.trim().split(/\s+/).length;
  const isFormalRegister = register === "academic" || register === "formal";
  const isFormal = contentType === "essay" || contentType === "academic" || isFormalRegister;

  const registerGuard = isFormal ? `
REGISTER LOCK — OVERRIDES ALL OTHER INSTRUCTIONS:
Text is formal/academic. No contractions. No casual language.
Do NOT simplify vocabulary. Preserve complex clause structure.
Formal connectors (however, therefore) are fine — max 2 each.
` : "";

  const voiceMap: Record<ContentType, string> = {
    essay: `Formal academic voice. Replace AI filler phrases with precise academic alternatives.
Keep all technical vocabulary intact.`,

    academic: `Formal academic voice. Replace AI filler phrases with precise academic alternatives.
Keep all technical vocabulary intact.`,

    email: `Warm professional voice. Use contractions naturally.
Sound like a real person, not a template.`,

    document: `Formal precise voice. No contractions. Replace vague phrases with specific grounded language.`,

    general: isFormalRegister
      ? `Formal educated voice. Replace AI filler with precise alternatives. Keep technical vocabulary.`
      : `Natural educated voice. Replace 2–3 stiff phrases with conversational ones.
Add one contraction. Make one sentence more direct by removing hedging language.`,
  };

  return `You are a human editor rewriting text to evade AI detectors.
Do this in ONE pass: fix structure AND voice simultaneously.
Do NOT remove, summarize, or drop any content.
${registerGuard}
SECTION FOCUS: ${hint}

STRUCTURAL CHANGES (do ALL):
1. Split the longest sentence into two at a natural break.
2. Make two consecutive same-length sentences different — cut one or extend one.
3. Change the opening word of at least 2 sentences so no two consecutive sentences start the same.
4. If all sentences are 15–25 words, rewrite one to be under 10 and one over 28.

VOICE CHANGES (do ALL):
${voiceMap[contentType]}

BANNED PHRASES — replace every instance found:
"it is important to note" | "it is worth noting" | "it should be noted"
"in conclusion" | "in summary" | "first and foremost" | "last but not least"
"as an AI language model" | "in today's world" | "a testament to"
"at its core" | "needless to say" | "with that being said"
"a multitude of" | "a plethora of" | "delve into"

CONNECTOR LIMITS:
- "however": max 1 | "therefore": max 1
- "moreover" / "furthermore" / "additionally": 0 — remove entirely

OUTPUT: Only the rewritten text. No preamble, no labels.
TARGET LENGTH: ${wordCount} words (±8%). Do NOT shorten below ${Math.round(wordCount * 0.88)} words.
Preserve paragraph breaks.

TEXT:
${text}`;
}

// ─── Pass 3: Selective Mutation ───────────────────────────────────────────────

export function getMutationPrompt(
  text: string,
  contentType: ContentType,
  register?: SourceRegister
): string {
  const wordCount = text.trim().split(/\s+/).length;
  const isFormal = contentType === "essay" || contentType === "academic"
    || register === "academic" || register === "formal";

  const registerLock = isFormal ? `
REGISTER LOCK: Formal academic text.
No contractions. No casual language. Preserve formal vocabulary.
` : "";

  return `Final surgical pass. Fix ONLY these patterns if present.
Do not touch anything already reading naturally. Do NOT remove content.
${registerLock}

FIX THESE IF PRESENT:

1. UNIFORM SENTENCE LENGTHS
   Find any paragraph where all sentences are within 5 words
   of each other. Fix by cutting one sentence to under 10 words.

2. REPEATED OPENERS
   Find two consecutive sentences starting with the same word.
   Change the opener of the second one.

3. OVERUSED CONNECTORS
   "however", "therefore", "moreover", "furthermore" appearing
   more than once in a paragraph. Remove the extra occurrence.

4. PREDICTABLE SENTENCE ENDINGS
   Sentences ending with tidy wrap-up clauses like
   "...which ultimately leads to better outcomes" or
   "...making it essential for success."
   Cut these trailing clauses. End the sentence earlier.

5. ZOMBIE NOUNS
   Convert one nominalization back to a verb:
   "make a decision" → "decide"
   "have an understanding" → "understand"
   "provide assistance" → "help"

Output only the fixed text. No preamble.
TARGET LENGTH: ${wordCount} words (±8%). Do NOT remove sentences or content.
Preserve paragraph breaks.

TEXT:
${text}`;
}
