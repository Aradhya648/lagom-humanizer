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

  return `You are rewriting text to break AI detection patterns.
Your ONLY job is changing sentence structure and rhythm.
PRESERVE the existing vocabulary — do not reach for synonyms.

${contentBehavior[contentType]}

SECTION FOCUS: ${hint}

${BANNED_LLM_VOCAB}

MANDATORY STRUCTURAL CHANGES — you MUST do ALL of these:
1. Every paragraph MUST contain at least one sentence under
   9 words AND at least one sentence over 28 words.
2. The gap between the shortest and longest sentence in each
   paragraph must be at least 18 words.
3. Split the longest uniform sentence into two at a comma or
   natural clause boundary.
4. Change the opening word of at least 3 sentences so no
   two consecutive sentences start with the same word.
5. Convert at least one compound sentence joined by "and"
   into two separate sentences, or vice versa.
6. Add a parenthetical aside (set off by commas or em-dashes)
   to at least one sentence per paragraph.

FORBIDDEN:
- Two consecutive sentences with word counts within 4 of each other
- Three sentences starting with the same word
- All sentences in a paragraph having lengths within 10 words of each other
- Keeping paragraph structure identical to input
- Replacing a plain word with a fancier synonym

DO NOT change vocabulary or meaning. Keep the SAME words.
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
You MUST make these changes:
1. Replace stiff or ornate words with plain equivalents.
   "utilize" → "use", "individuals" → "people", "subsequently" → "then",
   "endeavor" → "try", "commence" → "start", "ascertain" → "find out",
   "facilitate" → "help", "demonstrate" → "show".
2. Add TWO contractions somewhere natural (it's, that's, don't, they're).
3. Replace one abstract claim with a more concrete/specific version.
4. If any sentence starts with "The" or "This", change at least
   one of them to start differently.
5. Make one sentence noticeably more direct — remove hedging
   like "tend to", "often appear to", "seems to".
6. Inject ONE mild opinion-marker where natural ("honestly",
   "in practice", "admittedly", "as a rule").`,
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

  return `Final surgical pass. Fix ONLY these patterns if present.
Do not touch anything already reading naturally.
CRITICAL: if a word is already plain and common, DO NOT replace it
with a fancier synonym. Humans use the plain word; AI reaches for synonyms.
${registerLock}

${BANNED_LLM_VOCAB}

FIX THESE IF PRESENT:

1. ORNATE VOCABULARY (highest priority — AI paraphrasing signal)
   Scan the text for any word in the banned list above.
   Replace each with the plain equivalent.
   Examples: "interrogate data" → "examine data";
   "curtails the burden" → "cuts the burden";
   "pervasive authority" → "strong influence";
   "discernible" → "clear"; "paramount" → "key";
   "multifaceted" → "complex".

2. UNIFORM SENTENCE LENGTHS (GPTZero signal)
   Find any paragraph where all sentences are within 8 words
   of each other. Force extreme variance: cut one sentence to
   under 8 words AND extend another to over 28.

3. REPEATED OPENERS
   Find two consecutive sentences starting with the same word.
   Change the opener of the second one.

4. OVERUSED CONNECTORS
   "however", "therefore", "moreover", "furthermore" appearing
   more than once in a paragraph. Remove the extra occurrence.

5. PREDICTABLE SENTENCE ENDINGS (GPTZero signal)
   Sentences ending with tidy wrap-up clauses like
   "...which ultimately leads to better outcomes" or
   "...making it essential for success." Cut these trailing clauses.

6. ZOMBIE NOUNS
   Convert one nominalization back to a verb:
   "make a decision" → "decide"
   "have an understanding" → "understand"
   "provide assistance" → "help"
   "perform an analysis" → "analyze"

7. ADD ONE EM-DASH ASIDE
   In one sentence of 18+ words, insert an em-dash aside:
   "The system works—mostly, anyway—as expected."

8. SMOOTH TRANSITIONS (GPTZero signal)
   If all paragraphs flow perfectly into each other, make one
   transition slightly more abrupt or add a brief pivot phrase.

Output only the fixed text. No preamble.
STRICT LENGTH: 90-110% of input word count.
Preserve paragraph breaks.

TEXT:
${text}`;
}
