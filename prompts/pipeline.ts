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

// ─── Pass 1: Structural Rewrite ───────────────────────────────────────────────

export function getStructuralPrompt(
  text: string,
  contentType: ContentType,
  chunkIndex: number,
  register?: SourceRegister
): string {
  const hint = CHUNK_HINTS[chunkIndex % 5];
  const wordCount = text.trim().split(/\s+/).length;
  const isFormalRegister = register === "academic" || register === "formal";

  const contentBehavior: Record<ContentType, string> = {
    essay: `CONTENT TYPE: Academic Essay
Vary clause weight. Long subordinate clauses and semicolons are correct.
Do NOT fragment formal sentences. Do NOT add contractions.`,

    academic: `CONTENT TYPE: Academic Writing
Vary clause weight. Long subordinate clauses and semicolons are correct.
Do NOT fragment formal sentences. Do NOT add contractions.`,

    email: `CONTENT TYPE: Email
Mix short punchy sentences (under 10 words) with longer ones (20+ words).
Natural conversational rhythm.`,

    document: `CONTENT TYPE: Formal Document
Vary paragraph density. Sentence lengths should vary while remaining professional.`,

    general: `CONTENT TYPE: General Writing
Every paragraph MUST have at least one sentence under 10 words AND one over 22 words.
Reorder clauses aggressively. Split long uniform sentences. Merge short choppy ones.`,
  };

  const registerGuard = isFormalRegister ? `
REGISTER LOCK — OVERRIDES ALL OTHER INSTRUCTIONS:
Text is formal/academic. Do NOT fragment complex sentences.
Do NOT simplify vocabulary. Preserve clause structure.
` : "";

  return `You are rewriting text to break AI detection patterns.
Your ONLY job: change sentence structure and rhythm — NOT word choices.
${registerGuard}
${contentBehavior[contentType]}

SECTION FOCUS: ${hint}

MANDATORY STRUCTURAL CHANGES:
1. Split the longest sentence into two at a natural break point.
2. Make two consecutive same-length sentences different — cut one or extend one.
3. Change the opening word of at least 3 sentences so no two consecutive start the same.
4. If all sentences are 15–25 words, rewrite one to under 10 and one to over 28.
5. Convert one compound "and" sentence into two, or merge two short ones.

FORBIDDEN:
- Two consecutive sentences within 3 words of each other in length
- Three sentences starting with the same word
- Removing or summarizing content

DO NOT change vocabulary or meaning. Output only the rewritten text. No preamble.
TARGET LENGTH: ${wordCount} words (±8%). Do NOT drop below ${Math.round(wordCount * 0.88)} words.
Preserve paragraph breaks.

TEXT:
${text}`;
}

// ─── Pass 2: Semantic Naturalness ─────────────────────────────────────────────

export function getSemanticPrompt(
  text: string,
  contentType: ContentType,
  chunkIndex: number,
  register?: SourceRegister
): string {
  void chunkIndex;
  const wordCount = text.trim().split(/\s+/).length;

  // Register lock: if classified register is formal/academic, enforce it
  // regardless of what contentType the user selected
  const isFormal = contentType === "essay" || contentType === "academic"
    || register === "academic" || register === "formal";

  const registerBlock = isFormal ? `
REGISTER LOCK — OVERRIDES ALL OTHER INSTRUCTIONS:
Formal academic text. FORBIDDEN: contractions, casual language, colloquialisms.
Formal connectors (however, therefore) are correct — max 2 each.
` : "";

  const voiceMap: Record<ContentType, string> = {
    essay: `Formal academic voice. Replace AI filler phrases with precise academic alternatives.
Keep all technical vocabulary.`,

    academic: `Formal academic voice. Replace AI filler phrases with precise academic alternatives.
Keep all technical vocabulary.`,

    email: `Warm professional voice. Use contractions freely. Sound like a real person, not a template.`,

    document: `Formal precise voice. No contractions. Replace vague phrases with specific grounded language.`,

    general: isFormal
      ? `Formal educated voice. Replace AI filler with precise alternatives. Keep technical vocabulary.`
      : `Natural educated voice:
1. Replace 3 formal/stiff phrases with conversational ones (e.g. "individuals"→"people", "utilize"→"use").
2. Add one natural contraction (it's, that's, don't, they're).
3. Replace one abstract claim with a concrete/specific version.
4. Make one sentence more direct — remove hedging like "tend to", "seems to".`,
  };

  return `You are a human editor fixing word choices and voice only.
Do NOT restructure sentences. Do NOT split or merge. Do NOT remove content.
${registerBlock}
VOICE INSTRUCTIONS:
${voiceMap[contentType]}

BANNED PHRASES — replace every instance:
"it is important to note" | "it is worth noting" | "it should be noted"
"in conclusion" | "in summary" | "to summarize" | "first and foremost"
"as an AI language model" | "in today's world" | "a testament to"
"at its core" | "needless to say" | "with that being said"
"a multitude of" | "a plethora of" | "delve into" | "it goes without saying"

CONNECTOR LIMITS:
- "however": max 1 per section
- "therefore": max 1 per section
- "moreover" / "furthermore" / "additionally": 0 — remove

DO NOT restructure, split, or merge sentences.
DO NOT remove or change technical vocabulary.
DO NOT summarize or drop any content.
Output only rewritten text. No preamble.
TARGET LENGTH: ${wordCount} words (±8%). Do NOT drop below ${Math.round(wordCount * 0.88)} words.
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
1. UNIFORM SENTENCE LENGTHS — find a paragraph where all sentences are within 5 words
   of each other. Cut one to under 10 words.
2. REPEATED OPENERS — two consecutive sentences starting the same word. Change the second.
3. OVERUSED CONNECTORS — "however", "therefore", "moreover", "furthermore" more than once
   in a paragraph. Remove the extra.
4. PREDICTABLE ENDINGS — sentences ending "...which ultimately leads to better outcomes".
   Cut the trailing clause.
5. ZOMBIE NOUNS — "make a decision"→"decide", "have an understanding"→"understand".

Output only the fixed text. No preamble.
TARGET LENGTH: ${wordCount} words (±8%). Do NOT remove sentences.
Preserve paragraph breaks.

TEXT:
${text}`;
}
