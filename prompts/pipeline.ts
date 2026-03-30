export type HumanizeMode = "light" | "medium" | "aggressive";

// ─── Per-chunk variation hints ───────────────────────────────────────────────
// Applied by chunk index so adjacent chunks receive different micro-instructions.
// This introduces natural local inconsistency across the document.

const STRUCTURAL_HINTS = [
  "Open this section with a short, direct sentence.",
  "Let this section have at least one noticeably long sentence.",
  "Keep this section punchy — shorter sentences suit this part.",
  "This section can flow freely — longer sentences are fine here.",
  "Vary the rhythm here more than the surrounding sections.",
  "Make the first and last sentence of this section feel rhythmically different.",
];

const SEMANTIC_HINTS = [
  "The opening can be a touch more direct than usual.",
  "Concrete, specific language fits this section better than abstract phrasing.",
  "A slightly more relaxed turn of phrase is appropriate here.",
  "Keep this section crisp. No hedging, no softening.",
  "This section can be slightly more reflective in tone.",
  "The language here should feel natural and unpolished — not optimized.",
];

// ─── Pass 1: Structural Rewrite ──────────────────────────────────────────────
// Single responsibility: sentence rhythm and syntax only.
// Does NOT touch word choices, phrases, or meaning.
export function getStructuralPrompt(
  text: string,
  mode: HumanizeMode,
  chunkIndex?: number
): string {
  const scope: Record<HumanizeMode, string> = {
    light:
      "Minimal — find 2-3 sentences that are rhythmically identical to their neighbors and fix only those. Leave everything else alone.",
    medium:
      "Moderate — vary sentence lengths across the whole section. Aim for a clear mix: some under 12 words, some over 25. Break the uniform medium-length pattern.",
    aggressive:
      "Aggressive — no two consecutive sentences should share the same rhythmic shape (similar word count + similar structure). Vary every sentence against its neighbor.",
  };

  const hint =
    chunkIndex !== undefined
      ? `\nSECTION FOCUS: ${STRUCTURAL_HINTS[chunkIndex % STRUCTURAL_HINTS.length]}`
      : "";

  return `You are a structural copy editor. Your only task is sentence rhythm and syntax. Do not touch word choices, phrases, or meaning.

TASK: Restructure the sentences below so they no longer sound uniform and machine-generated.

SCOPE: ${scope[mode]}${hint}

WHAT TO DO:
- Vary sentence lengths drastically. AI writes all sentences at similar word counts. Break that — some sentences should be short (under 12 words), some long (25+).
- Split compound sentences that chain parallel clauses: "X does A, Y does B, and Z does C" — split or restructure these.
- Merge short choppy sentences that belong together into one longer flowing sentence.
- Vary sentence-opening structure. Never start two consecutive sentences with the same word or the same grammatical pattern (Subject-Verb, Subject-Verb, Subject-Verb is a red flag).
- If multiple paragraphs are the same visual weight, redistribute — make some dense, some light.

FORBIDDEN PATTERNS (these are AI tells — do not produce them):
- Two consecutive sentences with similar word counts (e.g. 22 words, then 21 words, then 23 words).
- Parallel list-like sentence triplets: "First, X. Second, Y. Third, Z."
- Compound sentences with symmetric halves: "While A does X, B does Y."
- Paragraphs where every sentence follows Subject + Verb + Object structure.
- Identical or near-identical sentence rhythms across an entire paragraph.

WHAT NOT TO DO:
- Do not change any vocabulary, word choices, or phrases unless structurally forced.
- Do not replace expressions, idioms, or terminology.
- Do not alter facts, arguments, or meaning in any way.
- Do not add ideas or remove ideas.

OUTPUT:
- Output only the rewritten text. No preamble, no labels, no meta-commentary.
- STRICT LENGTH RULE: output must be 90%–110% of the input word count. Do not elaborate. Do not pad. Do not expand.
- Preserve all paragraph breaks exactly.

TEXT:
${text}`;
}

// ─── Pass 2: Semantic Naturalness ────────────────────────────────────────────
// Runs after structure is fixed. Single responsibility: word choices + voice.
// Must NOT restructure sentences — rhythm is already set by Pass 1.
export function getSemanticPrompt(
  text: string,
  mode: HumanizeMode,
  chunkIndex?: number
): string {
  const depthMap: Record<HumanizeMode, string> = {
    light: `- Replace any obvious AI filler phrases from the list below with plain natural alternatives.
- Make one or two word-choice improvements that sound more like a person wrote them.
- Keep the professional tone intact.`,

    medium: `- Replace all AI filler phrases from the list below.
- Use contractions occasionally where they sound natural (it's, that's, don't, they're).
- Replace abstract or vague phrases with concrete, specific language.
- Add one or two connective phrases a real writer would reach for — not a machine.
- Allow mild imperfection: a slightly informal turn of phrase is better than a perfectly smooth one that reads like it was generated.
- Keep the register professional overall. Do not over-informalize.`,

    aggressive: `- Replace every AI filler phrase and formal connector from the list below.
- Use contractions freely where natural.
- Replace vague generalities with direct, specific, grounded language.
- Introduce natural imperfection: an occasional slight awkwardness, a compressed aside, a thought that ends a touch abruptly. Real writers do this. Machines don't.
- Where a sentence is over-polished or sounds editorial/academic, roughen it slightly.
- The writing should feel like someone who knows their subject sat down and wrote — not optimized, not neutral, not symmetric.
- Occasional abruptness is fine. A short blunt sentence after a long one is human.`,
  };

  const hint =
    chunkIndex !== undefined
      ? `\nSECTION FOCUS: ${SEMANTIC_HINTS[chunkIndex % SEMANTIC_HINTS.length]}`
      : "";

  return `You are a human editor focused on voice and word choice. Sentence structure has been set by a previous editor — do not change sentence boundaries or restructure syntax.

TASK: Make this text read like a real person wrote it. Focus only on word choices, transitions, and voice.${hint}

WHAT TO DO:
${depthMap[mode]}

BANNED PHRASES — remove or rewrite every instance that appears:
"it is important to note" | "it is worth noting" | "it is worth mentioning"
"it is crucial" | "it is essential" | "it is imperative" | "it should be noted"
"in conclusion" | "in summary" | "to summarize" | "this essay will"
"delve into" | "in today's world" | "in this day and age"
"plays a crucial role" | "plays a vital role"
"in the realm of" | "in the field of"
"as previously mentioned" | "as mentioned above"
"it can be argued" | "it is undeniable" | "it is clear that" | "it is evident that"
"needless to say" | "it goes without saying"
"in light of" | "taking into account" | "with regards to"

HARD LIMITS ON CONNECTOR WORDS (overuse is a primary AI detector signal):
- "however": use at most once in the entire section. Zero is fine.
- "therefore": use at most once. Prefer "so" or restructure the sentence.
- "moreover": use zero times. Find a concrete alternative.
- "furthermore": use zero times. Find a concrete alternative.
- "additionally": use zero times. Find a concrete alternative.
- "consequently": use zero times. Find a concrete alternative.
- "in conclusion": banned entirely.

FORBIDDEN PATTERNS — these are AI tells, do not produce them:
- Over-clean paragraph transitions that editorially summarize the previous point ("Having established X, we can now turn to Y...").
- Back-to-back sentences that are equally smooth and balanced — real writing has uneven texture.
- Pairs of sentences where the second perfectly mirrors the first in structure.
- Ending a section with a tidy wrap-up sentence that restates the main point.

WHAT NOT TO DO:
- Do not restructure sentences, split them, or merge them.
- Do not change argument order or paragraph structure.
- Do not add ideas or remove ideas.
- Do not make technical or professional content overly casual.
- Do not over-rewrite — this is refinement, not replacement.

OUTPUT:
- Output only the rewritten text. No preamble, no labels, no explanation.
- STRICT LENGTH RULE: output must be 90%–110% of the input word count. Do not elaborate or pad with filler.
- Preserve all paragraph breaks exactly.

TEXT:
${text}`;
}

// ─── Pass 3: Selective Mutation ──────────────────────────────────────────────
// Only fires when the internal detector still scores the text as synthetic
// after Passes 1 and 2. Surgical — targets specific statistical patterns.
export function getMutationPrompt(text: string): string {
  return `You are doing a final surgical pass on text that has already been edited twice. Most of it is now fine. Your job is narrow: identify and fix only the specific patterns that AI detectors flag.

TASK: Find and fix only the spots listed below. Do not touch anything that already reads naturally.

WHAT AI DETECTORS MEASURE — fix any of these that remain:

1. SENTENCE LENGTH UNIFORMITY
   All sentences in a paragraph are similar lengths (e.g. 18, 20, 22, 19 words).
   Fix: Break one or two sentences dramatically shorter or longer to widen the variance.

2. CONNECTOR OVERUSE
   "however", "therefore", "moreover", "furthermore", "additionally", "consequently"
   appearing more than once total across the section.
   Fix: Remove the excess ones or restructure the sentences so they're not needed.

3. CONSECUTIVE SAME OPENERS
   Two or more sentences in a row starting with the same word.
   Fix: Reopen one of them differently.

4. OVER-SMOOTH PARAGRAPH STITCHING
   A transition sentence between paragraphs that feels editorial — like a narrator
   stitching sections together rather than a writer continuing a thought.
   Fix: Remove or rough it up. An abrupt paragraph start is more human.

5. CLAUSE LENGTH UNIFORMITY
   Comma-separated clauses that are all similar lengths inside sentences.
   Fix: Break the symmetry — make one clause very short, another very long.

6. PARALLEL SENTENCE GROUPS
   Two or three consecutive sentences that follow the exact same grammatical structure.
   Fix: Restructure one of them so it breaks the pattern.

HARD RULE:
If a sentence already reads naturally, do not touch it. The goal is to fix statistical
outliers, not to rewrite clean text. Leave good writing alone.

OUTPUT:
- Output only the final text. No preamble, no commentary, no labels.
- STRICT LENGTH RULE: output must be 90%–110% of the input word count. Do not elaborate or pad with filler.
- Preserve all paragraph breaks exactly.

TEXT:
${text}`;
}
