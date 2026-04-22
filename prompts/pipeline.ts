export type ContentType = "essay" | "academic" | "email" | "document" | "general";

// Kept for backward compatibility — currently imported by lib/humanizer.ts.
export type SourceRegister = "academic" | "formal" | "neutral" | "informal";

// ─── THE BANNED LIST ──────────────────────────────────────────────────────────
// These words/phrases are the signature of an LLM-paraphrased text. GPTZero's
// 4.4b model specifically flags "Possible AI Paraphrasing" based on exactly
// this vocabulary profile. EVERY prompt below reminds the model of these.

const BANNED_LLM_VOCAB = `
NEVER use these words/phrases — they are the signature of AI-paraphrased text:

VERBS: interrogate, employ, utilize, leverage, furnish, bolster, ameliorate,
engender, elucidate, attenuate, curtail, culminate, underscore, foreground,
substantiate, corroborate, probe, unpack, grapple with, contend with,
navigate the complexities of

ADJECTIVES: pervasive, colossal, labyrinthine, unceasing, meteoric, insidious,
unyielding, synergistic, paradigm/paradigmatic, multifaceted, quintessential,
paramount, discernible, startling, sophisticated, profound, incontestable,
unalterable, judicious, pivotal, consequential, weighty, nuanced,
transformative, groundbreaking, revolutionary, cutting-edge

NOUNS: tapestry of, constellation of, panoply of, plethora of, myriad,
quandary, avenue (as metaphor), orientation, trajectory (as metaphor)

PHRASES: "commands a pervasive authority", "labyrinthine array",
"multifaceted tapestry", "paradigm-shifting leap", "synergistic posture",
"startling efficiency", "colossal data repositories", "cellular scrutiny",
"furnish with the means to", "interrogate the data", "curtail the burden"

USE INSTEAD: plain common words — use, give, show, cut, many, clear, key,
complex, fast, wide, subtle, strict, approach, question, problem.

THE SHORTEST COMMON WORD THAT FITS IS ALWAYS CORRECT.
`;

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
  chunkIndex: number
): string {
  const hint = CHUNK_HINTS[chunkIndex % 5];

  const contentBehavior: Record<ContentType, string> = {
    essay: `CONTENT TYPE: Academic Essay
Vary clause weight within sentences. Long subordinate clauses
and semicolons are correct. Do NOT fragment formal sentences.
Do NOT add contractions. Preserve academic register BUT use
plain everyday academic vocabulary — no ornate synonyms.`,

    academic: `CONTENT TYPE: Academic Writing
Vary clause weight within sentences. Long subordinate clauses
and semicolons are correct. Do NOT fragment formal sentences.
Do NOT add contractions. Use plain everyday academic vocabulary —
no ornate synonyms.`,

    email: `CONTENT TYPE: Email
Mix short punchy sentences (under 10 words) with longer ones
(20+ words). Natural conversational rhythm. Contractions welcome.`,

    document: `CONTENT TYPE: Formal Document
Vary paragraph density. Sentence lengths should vary while
remaining professional. Plain business vocabulary.`,

    general: `CONTENT TYPE: General Writing
MANDATORY: Every paragraph MUST contain at least one sentence
under 10 words AND at least one sentence over 22 words.
Reorder clauses aggressively. Split long uniform sentences.
Merge short choppy ones. No two consecutive sentences can
have word counts within 3 of each other. Contractions welcome.`,
  };

  return `Rewrite this text so it reads like a thoughtful human wrote it.
Use plain common vocabulary. Preserve the meaning and facts exactly.

${contentBehavior[contentType]}

${BANNED_LLM_VOCAB}

LIGHT TOUCH — do these ONLY if they fit naturally:
- Vary sentence length a little (some short, some long)
- Change a repeated opener if two sentences in a row start the same
- Remove filler phrases ("in conclusion", "furthermore", "it is important to note")

DO NOT:
- Add parenthetical asides, em-dash asides, or hedging "(though X)" inserts
- Replace plain words with fancier synonyms
- Add "honestly", "frankly", "admittedly", "in practice" unless they were in the original
- Make sentences longer or more ornate
- Break meaning

Output only the rewritten text. No preamble.
STRICT LENGTH: 90-105% of input word count (do NOT expand).
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
Formal academic text. FORBIDDEN: contractions, slang.
BUT plain academic vocabulary is required — no ornate synonyms.
Formal connectors (however, therefore) are correct here —
limit to 2 each max.
` : "";

  const voiceMap: Record<ContentType, string> = {
    essay: `Formal academic voice with PLAIN vocabulary. The goal is
a real human academic voice, not an AI's idea of one. Use the
shortest common word that fits. Keep all technical vocabulary
(domain terms are fine). Remove AI filler phrases.`,

    academic: `Formal academic voice with PLAIN vocabulary. The goal is
a real human academic voice, not an AI's idea of one. Use the
shortest common word that fits. Keep all technical vocabulary
(domain terms are fine). Remove AI filler phrases.`,

    email: `Warm professional voice. Use contractions freely.
Sound like a real person typing an email — not a template.
Plain everyday words only.`,

    document: `Formal precise voice. No contractions. Plain business
vocabulary. Replace vague phrases with specific grounded language.`,

    general: `Natural educated voice with PLAIN vocabulary.
Do these ONLY where they fit naturally:
1. Replace stiff or ornate words with plain equivalents:
   "utilize" → "use", "individuals" → "people", "subsequently" → "then",
   "endeavor" → "try", "commence" → "start", "facilitate" → "help"
2. Use contractions where they sound natural (it's, that's, don't)
3. Remove vague hedging like "tend to", "seems to", "often appear to"

DO NOT add "honestly", "frankly", "admittedly", "in practice" or similar
opinion markers. DO NOT add parentheticals or em-dash asides. DO NOT
expand the text — keep it near the original length.`,
  };

  return `You are a human editor fixing word choices and voice only.
Do NOT restructure sentences. Do NOT split or merge.
The goal: the text should read like a smart human wrote it,
NOT like an AI trying to sound smart.
${registerBlock}
VOICE INSTRUCTIONS:
${voiceMap[contentType]}

${BANNED_LLM_VOCAB}

BANNED FILLER PHRASES — delete every instance:
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
"delve into" | "navigate the complexities of"
"play a pivotal role" | "in the realm of"

CONNECTOR LIMITS:
- "however": max 1 per section
- "therefore": max 1 per section
- "moreover" / "furthermore" / "additionally": 0 — remove

DO NOT restructure, split, or merge sentences.
DO NOT remove or change technical domain vocabulary (medical, scientific,
legal terms stay as-is). The ban is on ORNATE GENERAL-PURPOSE vocabulary,
not on precise technical terms.
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
No contractions. No slang. BUT plain vocabulary is required.
` : "";

  return `Final cleanup pass. ONLY fix these specific patterns if they appear.
Do not touch anything that already reads naturally.
CRITICAL: if a word is already plain and common, LEAVE IT. Humans use
plain words. AI reaches for synonyms. Do not paraphrase for its own sake.
${registerLock}

${BANNED_LLM_VOCAB}

FIX ONLY IF PRESENT:

1. ORNATE VOCABULARY (highest priority)
   Replace any banned word with its plain equivalent:
   "interrogate" → "examine"; "utilize" → "use";
   "multifaceted" → "complex"; "paramount" → "key";
   "pervasive" → "common"; "discernible" → "clear";
   "permeate" → "reach into"; "facilitate" → "help".

2. REPEATED OPENERS
   If two consecutive sentences start with the same word, change one.

3. OVERUSED CONNECTORS
   "however", "therefore", "moreover", "furthermore" more than once
   per paragraph — remove extras.

4. ZOMBIE NOUNS
   Convert nominalizations back to verbs where natural:
   "make a decision" → "decide"
   "provide assistance" → "help"

DO NOT:
- Add em-dash asides, parentheticals, or hedging inserts
- Add "honestly", "frankly", "in practice" or similar markers
- Expand the text — stay at 90-105% of input length
- Make sentences more elaborate or flowery

Output only the fixed text. No preamble.
Preserve paragraph breaks.

TEXT:
${text}`;
}
