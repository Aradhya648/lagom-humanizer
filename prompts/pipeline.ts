export type ContentType = "essay" | "academic" | "email" | "document" | "general";

// Kept for backward compatibility — currently imported by lib/humanizer.ts.
// Remove when humanizer.ts is updated to use ContentType.
export type SourceRegister = "academic" | "formal" | "neutral" | "informal";

// ─── Chunk Variation Hints ────────────────────────────────────────────────────
// Cycled by chunkIndex % 5 inside getStructuralPrompt.

const CHUNK_HINTS = [
  "Open with a short direct sentence under 10 words.",
  "Let one sentence in this section run long — over 28 words.",
  "Keep this section punchy. Shorter sentences throughout.",
  "Vary the first word of every sentence — no two the same.",
  "Make the last sentence of this section shorter than the rest.",
];

// ─── Pass 1: Structural Rewrite ───────────────────────────────────────────────

export function getStructuralPrompt(
  text: string,
  contentType: ContentType,
  chunkIndex: number
): string {
  const hint = CHUNK_HINTS[chunkIndex % 5];

  const contentBehavior: Record<ContentType, string> = {
    essay: `CONTENT TYPE: Academic Essay
Vary clause weight within sentences. Long subordinate clauses
and semicolons are correct. Do NOT fragment formal sentences.
Do NOT add contractions. Preserve academic register.`,

    academic: `CONTENT TYPE: Academic Writing
Vary clause weight within sentences. Long subordinate clauses
and semicolons are correct. Do NOT fragment formal sentences.
Do NOT add contractions. Preserve academic register.`,

    email: `CONTENT TYPE: Email
Mix short punchy sentences (under 10 words) with longer ones
(20+ words). Natural conversational rhythm.`,

    document: `CONTENT TYPE: Formal Document
Vary paragraph density. Sentence lengths should vary while
remaining professional.`,

    general: `CONTENT TYPE: General Writing
MANDATORY: Every paragraph MUST contain at least one sentence
under 10 words AND at least one sentence over 22 words.
Reorder clauses aggressively. Split long uniform sentences.
Merge short choppy ones. No two consecutive sentences can
have word counts within 3 of each other.`,
  };

  return `You are rewriting text to break AI detection patterns.
Your ONLY job is changing sentence structure and rhythm —
not word choices.

${contentBehavior[contentType]}

SECTION FOCUS: ${hint}

MANDATORY STRUCTURAL CHANGES — you MUST do ALL of these:
1. Identify the longest sentence and split it into two at a
   natural break point.
2. Identify two consecutive sentences with similar lengths
   and make one significantly shorter (cut it by half) or
   significantly longer (add a dependent clause).
3. Change the opening word of at least 3 sentences so no
   two consecutive sentences start with the same word.
4. If any paragraph has 4+ sentences all between 15-25 words,
   rewrite one to be under 10 words and one to be over 28.
5. Convert at least one compound sentence joined by "and"
   into two separate sentences, or vice versa.

FORBIDDEN:
- Two consecutive sentences with word counts within 3 of each other
- Three sentences starting with the same word
- Keeping paragraph structure identical to input

DO NOT change vocabulary or meaning.
Output only the rewritten text. No preamble.
STRICT LENGTH: 90-110% of input word count.
Preserve paragraph breaks.

TEXT:
${text}`;
}

// ─── Pass 2: Semantic Naturalness ─────────────────────────────────────────────

export function getSemanticPrompt(
  text: string,
  contentType: ContentType,
  chunkIndex: number
): string {
  void chunkIndex;

  const isFormal = contentType === "essay" || contentType === "academic";

  const registerBlock = isFormal ? `
REGISTER LOCK — OVERRIDES ALL OTHER INSTRUCTIONS
Formal academic text. FORBIDDEN: contractions, casual language,
colloquialisms. Formal connectors (however, therefore) are
correct here — limit to 2 each max.
` : "";

  const voiceMap: Record<ContentType, string> = {
    essay: `Formal academic voice. Replace AI filler phrases with
precise academic alternatives. Keep all technical vocabulary.`,

    academic: `Formal academic voice. Replace AI filler phrases with
precise academic alternatives. Keep all technical vocabulary.`,

    email: `Warm professional voice. Use contractions freely.
Sound like a real person, not a template.`,

    document: `Formal precise voice. No contractions. Replace vague
phrases with specific grounded language.`,

    general: `Natural educated voice. You MUST make these changes:
1. Replace at least 3 formal/stiff phrases with conversational
   equivalents (e.g. "individuals" → "people", "utilize" → "use",
   "subsequently" → "then", "endeavor" → "try").
2. Add one contraction somewhere natural (it's, that's, don't,
   they're, we've).
3. Replace one abstract claim with a more concrete/specific version.
4. If any sentence starts with "The" or "This", change at least
   one of them to start differently.
5. Make one sentence noticeably more direct — remove hedging
   language like "tend to", "often appear to", "seems to".`,
  };

  return `You are a human editor fixing word choices and voice only.
Do NOT restructure sentences. Do NOT split or merge.
${registerBlock}
VOICE INSTRUCTIONS:
${voiceMap[contentType]}

BANNED PHRASES — replace every instance:
"it is important to note" | "it is worth noting"
"it is worth mentioning" | "it should be noted"
"in conclusion" | "in summary" | "to summarize"
"this essay will" | "as an AI language model"
"in today's fast-paced world" | "first and foremost"
"last but not least" | "all in all" | "in a nutshell"
"it goes without saying" | "needless to say"
"a multitude of" | "a plethora of"
"in today's world" | "a testament to"
"at its core" | "it is no secret that"
"with that being said" | "that being said"

CONNECTOR LIMITS:
- "however": max 1 per section
- "therefore": max 1 per section
- "moreover" / "furthermore" / "additionally": 0 — remove

DO NOT restructure, split, or merge sentences.
DO NOT remove or change technical vocabulary.
Output only rewritten text. No preamble.
STRICT LENGTH: 90-110% of input word count.
Preserve paragraph breaks.

TEXT:
${text}`;
}

// ─── Pass 3: Selective Mutation ───────────────────────────────────────────────

export function getMutationPrompt(
  text: string,
  contentType: ContentType
): string {
  const isFormal = contentType === "essay" || contentType === "academic";

  const registerLock = isFormal ? `
REGISTER LOCK: Formal academic text.
No contractions. No casual language. Preserve formal vocabulary.
` : "";

  return `Final surgical pass. Fix ONLY these patterns if present.
Do not touch anything already reading naturally.
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
   "give consideration" → "consider"
   "reach a conclusion" → "conclude"

Output only the fixed text. No preamble.
STRICT LENGTH: 90-110% of input word count.
Preserve paragraph breaks.

TEXT:
${text}`;
}
