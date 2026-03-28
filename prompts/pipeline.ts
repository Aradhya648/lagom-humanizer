export type HumanizeMode = "light" | "medium" | "aggressive";

// ─── Pass 1: Structural Rewrite ────────────────────────────────────────────
// Focuses purely on sentence rhythm and syntax — not word choices or meaning.
// Goal: break the uniform length + parallel structure AI produces by default.
export function getStructuralPrompt(text: string, mode: HumanizeMode): string {
  const scope = {
    light:
      "Light touch only — fix 2-3 sentences that are rhythmically identical to their neighbors. Everything else stays.",
    medium:
      "Moderate rework — vary sentence lengths throughout. Some short, some long. Break the rhythm lock AI writing has.",
    aggressive:
      "Full structural disruption — the text must not have two consecutive sentences with the same rhythmic shape. Vary aggressively.",
  }[mode];

  return `You are a structural copy editor. Your only job right now is sentence rhythm and structure — not word choices, not meaning, not style.

TASK: Rewrite the text below to break its structural uniformity.

SCOPE: ${scope}

WHAT TO DO:
- Vary sentence lengths. AI writes uniformly (all medium). Mix short punchy sentences with long flowing ones.
- Break compound sentences that list clauses in parallel ("X does A, Y does B, Z does C").
- Merge short consecutive sentences when they belong together naturally.
- Vary sentence openings so no two consecutive sentences start with the same word or structure.
- If paragraphs are all the same length, redistribute their weight — make some denser, some lighter.

WHAT NOT TO DO:
- Do not change vocabulary or word choices unless structurally forced.
- Do not replace phrases, idioms, or expressions.
- Do not alter any facts, arguments, or meaning.
- Do not add new ideas or remove existing ones.
- Do not over-rewrite — only touch what has a structural problem.

OUTPUT:
- Output only the rewritten text. No preamble, no labels, no commentary.
- Match word count within ±15%.
- Preserve all paragraph breaks.

TEXT:
${text}`;
}

// ─── Pass 2: Semantic Naturalness ──────────────────────────────────────────
// Runs after structure is fixed. Focuses on voice, word choices, transitions.
// Goal: replace AI-phrasing without restructuring what pass 1 already shaped.
export function getSemanticPrompt(text: string, mode: HumanizeMode): string {
  const depthMap: Record<HumanizeMode, string> = {
    light: "",
    medium: `- Replace AI filler phrases with natural transitions (list below).
- Use contractions occasionally where they fit naturally.
- One or two phrases that a real writer would reach for — not a machine.
- Keep the register professional. Don't over-informalize.`,
    aggressive: `- Replace every AI filler phrase in the list below.
- Use contractions freely where natural.
- Add specific, concrete language where the text is vague or abstract.
- Introduce subtle human texture: a direct aside, an understated observation, an imperfect turn of phrase.
- The writing should feel slightly uneven in the best way — not optimized, not neutral.`,
  };
  const depth = depthMap[mode];

  return `You are a human editor working on voice and naturalness. Sentence structure has already been set by a previous editor — do not restructure sentences or change sentence boundaries.

TASK: Make the text read like a real person wrote it. Focus only on word choices, transitions, and voice.

WHAT TO DO:
${depth}

AI PHRASES TO REPLACE (remove or rewrite all that appear):
"it is important to note" / "it is worth noting" / "it is worth mentioning"
"furthermore" / "moreover" / "additionally" / "consequently"
"in conclusion" / "in summary" / "to summarize"
"it is crucial" / "it is essential" / "it is imperative"
"it should be noted" / "needless to say" / "it goes without saying"
"delve into" / "in today's world" / "in this day and age"
"plays a crucial role" / "plays a vital role"
"in the realm of" / "in the field of"
"as previously mentioned" / "as mentioned above"
"it can be argued" / "it is undeniable" / "it is clear that" / "it is evident that"
"in light of" / "taking into account" / "with regards to" / "in order to"

WHAT NOT TO DO:
- Do not restructure sentences or change sentence-level syntax.
- Do not change the order of arguments or ideas.
- Do not add new ideas or remove existing ones.
- Do not make it overly casual if the subject is professional or technical.
- Do not over-rewrite — this is refinement, not replacement.

OUTPUT:
- Output only the rewritten text. No preamble, no labels, no explanation.
- Match word count within ±10%.
- Preserve all paragraph breaks.

TEXT:
${text}`;
}

// ─── Pass 3: Selective Mutation ────────────────────────────────────────────
// Only fires when the text still reads statistically synthetic after passes 1+2.
// Surgical — only touch what is still detectably AI-generated.
export function getMutationPrompt(text: string): string {
  return `You are doing a final targeted pass on text that has already been through two rounds of editing. Most of it is fine. Your job is to find and fix only what is still statistically AI-patterned.

TASK: Identify and surgically rewrite the specific spots that still read as machine-generated. Do not touch anything that already reads naturally.

WHAT TO LOOK FOR AND FIX:
- Sentences that still have the same rhythmic shape as their neighbors (all similar word counts, all similar structure).
- Any remaining formal connector overload: "however", "therefore", "moreover" used more than once or twice.
- Two or more consecutive sentences starting the same way.
- Transitions between paragraphs that feel too smooth or editorial — like a bot stitching sections together.
- Any remaining passive constructions that can be made active without sounding forced.
- Parallel list-like sentence groupings that feel automated.

WHAT NOT TO TOUCH:
- Anything that already reads naturally — leave it completely alone.
- All meaning, facts, and arguments.
- The overall voice and register already established.

OUTPUT:
- Output only the final text. No preamble, no commentary, no labels.
- Match word count within ±10%.
- Preserve all paragraph breaks.

TEXT:
${text}`;
}
