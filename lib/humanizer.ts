import { GoogleGenerativeAI } from "@google/generative-ai";
import { detectAI } from "@/lib/detector";
import {
  getStructuralPrompt,
  getSemanticPrompt,
  getMutationPrompt,
  type HumanizeMode,
} from "@/prompts/pipeline";

export type { HumanizeMode };

// ─── Utilities ──────────────────────────────────────────────────────────────

function truncateToWordLimit(text: string, wordLimit: number): string {
  const words = text.trim().split(/\s+/);
  if (words.length <= wordLimit) return text.trim();
  return words.slice(0, wordLimit).join(" ");
}

// Split a paragraph into its individual sentences.
function getSentences(para: string): string[] {
  return para
    .split(/(?<=[.!?])\s+(?=[A-Z"'])/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

// Variable-size chunker.
//
// Strategy:
//   1. Split on paragraph boundaries first.
//   2. Short paragraphs (≤3 sentences) → kept as a single chunk.
//   3. Longer paragraphs → split into variable-size sentence groups (2 or 3
//      sentences each) so adjacent chunks have different rhythmic weights.
//   4. Tiny chunks (< 15 words) are merged into the previous chunk.
//   5. Total chunk count capped at 8 to keep API calls bounded.
//
// Goal: introduce natural local inconsistency — humans write paragraph-by-
// paragraph with uneven focus, not with global uniformity.
export function splitIntoVariableChunks(text: string): string[] {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const raw: string[] = [];

  for (const para of paragraphs) {
    const sentences = getSentences(para);

    if (sentences.length <= 3) {
      // Short paragraph — keep whole
      raw.push(para);
      continue;
    }

    // Longer paragraph — split into variable sentence groups
    let i = 0;
    while (i < sentences.length) {
      const wordCount = sentences[i].split(/\s+/).length;
      // Long sentence → group of 2; short sentence → alternate 2/3
      const groupSize =
        wordCount > 22 ? 2 : i % 2 === 0 ? 3 : 2;
      const end = Math.min(i + groupSize, sentences.length);
      raw.push(sentences.slice(i, end).join(" "));
      i = end;
    }
  }

  // Merge chunks that are too short (< 15 words) into predecessor
  const chunks: string[] = [];
  for (const chunk of raw) {
    if (chunks.length > 0 && chunk.split(/\s+/).length < 15) {
      chunks[chunks.length - 1] += " " + chunk;
    } else {
      chunks.push(chunk);
    }
  }

  // Cap at 8 chunks — merge tail overflow into last chunk
  if (chunks.length > 8) {
    const head = chunks.slice(0, 7);
    const tail = chunks.slice(7).join(" ");
    return [...head, tail];
  }

  return chunks.length > 0 ? chunks : [text.trim()];
}

// ─── Sentence Classification ─────────────────────────────────────────────────
// Classifies each sentence as:
//   A = plain factual (low AI signal → minimal rewrite)
//   B = medium synthetic (moderate AI signal → moderate rewrite)
//   C = high-risk synthetic (strong AI signal → full rewrite)
//
// Classification heuristics based on what GPTZero/Originality.ai weight:
//   - Filler phrases → C
//   - Formal connectors at sentence start → C
//   - Very uniform sentence length matching neighbors → B
//   - Short, direct, factual statements → A
//   - Sentences under 10 words → A (too short to carry AI signal)

const SENTENCE_FILLER_RE = /\b(it is important to note|it is worth noting|furthermore|moreover|additionally|consequently|in conclusion|it is crucial|it is essential|delve into|in today's world|plays a crucial role|plays a vital role)\b/i;
const SENTENCE_FORMAL_OPENER_RE = /^(However|Moreover|Furthermore|Additionally|Consequently|Nevertheless|It is important|It is worth|In conclusion|In summary)\b/i;

export type SentenceClass = "A" | "B" | "C";

export function classifySentence(
  sentence: string,
  avgNeighborLength?: number
): SentenceClass {
  const words = sentence.split(/\s+/).length;

  // Very short sentences → plain factual, skip rewrite
  if (words <= 10) return "A";

  // Contains filler phrases → high-risk
  if (SENTENCE_FILLER_RE.test(sentence)) return "C";

  // Starts with formal connector → high-risk
  if (SENTENCE_FORMAL_OPENER_RE.test(sentence)) return "C";

  // Uniform length relative to neighbors → medium risk
  if (avgNeighborLength !== undefined) {
    const ratio = words / avgNeighborLength;
    if (ratio > 0.85 && ratio < 1.15) return "B"; // suspiciously uniform
  }

  // Short-to-medium plain statements → keep
  if (words <= 16) return "A";

  // Default medium
  return "B";
}

// Classify all sentences in a chunk and produce annotated text for the prompt.
export function annotateChunkWithClasses(chunk: string): {
  annotated: string;
  classes: SentenceClass[];
} {
  const sentences = getSentences(chunk);
  if (sentences.length === 0) return { annotated: chunk, classes: [] };

  const lengths = sentences.map((s) => s.split(/\s+/).length);
  const classes: SentenceClass[] = sentences.map((s, i) => {
    // Avg length of neighbors (excluding self)
    const neighbors = lengths.filter((_, j) => j !== i && Math.abs(j - i) <= 1);
    const avgNeighbor =
      neighbors.length > 0
        ? neighbors.reduce((a, b) => a + b, 0) / neighbors.length
        : undefined;
    return classifySentence(s, avgNeighbor);
  });

  // Build annotated text with inline tags
  const annotated = sentences
    .map((s, i) => `[${classes[i]}] ${s}`)
    .join("\n");

  return { annotated, classes };
}

// ─── Generation Settings ─────────────────────────────────────────────────────
// Each pass uses a distinct sampling regime so pass 2 cannot statistically
// mirror pass 1's output distribution.

interface GenSettings {
  temperature: number;
  topP: number;
  topK?: number;           // undefined = unconstrained (max variety)
  frequencyPenalty?: number; // 0–1: penalises repeating tokens seen so far
  presencePenalty?: number;  // 0–1: penalises tokens already present at all
}

// Structural pass: maximum syntactic diversity.
// High topP + no topK + frequencyPenalty → broad sampling, discourages
// repeating the same sentence patterns within a chunk.
const STRUCTURAL_SETTINGS: Record<HumanizeMode, GenSettings> = {
  light:      { temperature: 0.72, topP: 0.95, frequencyPenalty: 0.15 },
  medium:     { temperature: 0.88, topP: 0.97, frequencyPenalty: 0.25 },
  aggressive: { temperature: 0.97, topP: 0.97, frequencyPenalty: 0.30, presencePenalty: 0.10 },
};

// Semantic pass: focused natural vocabulary.
// topK=40 constrains word-choice to common/natural tokens, preventing the
// model from drifting back into unusual AI-typical phrasings.
// Lower topP + light penalty → stable, grounded language.
const SEMANTIC_SETTINGS: Record<HumanizeMode, GenSettings> = {
  light:      { temperature: 0.70, topP: 0.88, topK: 40 },
  medium:     { temperature: 0.73, topP: 0.88, topK: 40, frequencyPenalty: 0.10 },
  aggressive: { temperature: 0.80, topP: 0.90, topK: 40, frequencyPenalty: 0.12 },
};

// Mutation pass: targeted and precise.
// Medium temperature + topK=35 → focused changes without random drift.
const MUTATION_SETTINGS: GenSettings = {
  temperature: 0.88,
  topP: 0.92,
  topK: 35,
  frequencyPenalty: 0.20,
  presencePenalty: 0.08,
};

// ─── Model Wrappers ─────────────────────────────────────────────────────────

async function callGemini(prompt: string, settings: GenSettings): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-flash",
    generationConfig: {
      temperature: settings.temperature,
      topP: settings.topP,
      topK: settings.topK,
      frequencyPenalty: settings.frequencyPenalty,
      presencePenalty: settings.presencePenalty,
      maxOutputTokens: 4096,
    },
  });

  const result = await model.generateContent(prompt);
  const text = result.response.text();
  if (!text || text.trim().length === 0) {
    throw new Error("Gemini returned empty response");
  }
  return text.trim();
}

// Emergency fallback — only used when Gemini fails entirely.
// Not used for per-chunk retries; that would obscure the root cause.
async function callHuggingFace(prompt: string): Promise<string> {
  const apiKey = process.env.HUGGINGFACE_API_KEY;
  const HF_API_URL =
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2";

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  const response = await fetch(HF_API_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({
      inputs: `<s>[INST] ${prompt} [/INST]`,
      parameters: {
        max_new_tokens: 2048,
        temperature: 0.85,
        top_p: 0.95,
        do_sample: true,
        return_full_text: false,
      },
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`HuggingFace API error ${response.status}: ${err}`);
  }

  const data = await response.json();
  if (Array.isArray(data) && data[0]?.generated_text) {
    return data[0].generated_text.trim();
  }
  throw new Error("Unexpected HuggingFace response format");
}

// Primary: Gemini with pass-specific settings. Fallback: HuggingFace (Gemini failure only).
async function callModel(prompt: string, settings: GenSettings): Promise<string> {
  try {
    return await callGemini(prompt, settings);
  } catch (geminiError) {
    console.warn("Gemini failed, falling back to HuggingFace:", geminiError);
    return await callHuggingFace(prompt);
  }
}

// ─── Pipeline Passes ────────────────────────────────────────────────────────

// Validate a model output for a chunk: must be non-empty.
// If the model returns empty, fall back to the original input for that chunk.
function validateChunkOutput(output: string, fallback: string): string {
  return output.trim().length > 0 ? output.trim() : fallback;
}

// Pass 1 — Structural rewrite per chunk.
// Parallel across chunks. Each chunk gets a variation hint via chunkIndex.
// Uses high-diversity settings: unconstrained topK, high topP, frequencyPenalty.
// Per-chunk failures fall back to the original chunk text — a single bad API
// call should not abort the entire request.
// Phase 13: Annotates each chunk with sentence classifications (A/B/C) so the
// LLM knows which sentences to preserve vs. rewrite aggressively.
async function structuralPass(
  chunks: string[],
  mode: HumanizeMode
): Promise<string[]> {
  const settings = STRUCTURAL_SETTINGS[mode];
  return Promise.all(
    chunks.map((chunk, i) => {
      const { annotated } = annotateChunkWithClasses(chunk);
      return callModel(
        getStructuralPrompt(chunk, mode, i, annotated),
        settings
      )
        .then((out) => validateChunkOutput(out, chunk))
        .catch(() => chunk);
    })
  );
}

// Pass 2 — Semantic naturalness per chunk.
// Parallel across chunks. Uses topK=40 to constrain word-choice to natural
// human vocabulary — statistically distinct from the structural pass regime.
// Skipped entirely for light mode.
async function semanticPass(
  chunks: string[],
  mode: HumanizeMode
): Promise<string[]> {
  if (mode === "light") return chunks;
  const settings = SEMANTIC_SETTINGS[mode];
  return Promise.all(
    chunks.map((chunk, i) =>
      callModel(getSemanticPrompt(chunk, mode, i), settings)
        .then((out) => validateChunkOutput(out, chunk))
        .catch(() => chunk) // fall back to structural output on any failure
    )
  );
}

// Pass 3 — Selective mutation on the full merged text.
// Runs once on the assembled output, not per chunk — short chunks give
// noisy detector readings. Gated by score threshold; skipped if already good.
async function mutationPass(
  text: string,
  mode: HumanizeMode
): Promise<string> {
  if (mode === "light") return text;

  const threshold = mode === "aggressive" ? 42 : 52;
  const { score } = detectAI(text);

  if (score <= threshold) return text; // already good — skip

  return callModel(getMutationPrompt(text), MUTATION_SETTINGS);
}

// ─── First-Paragraph Hardening ───────────────────────────────────────────────
// Detectors weight the first 150-200 words most heavily. If the opening
// paragraph still scores high after all passes, give it an extra targeted
// mutation round. This adds at most 1 LLM call.

async function firstParagraphHardening(
  text: string,
  mode: HumanizeMode
): Promise<string> {
  if (mode === "light") return text;

  const paragraphs = text.split(/\n\s*\n/);
  if (paragraphs.length === 0) return text;

  const firstPara = paragraphs[0];
  const { score } = detectAI(firstPara);

  // Threshold: first paragraph needs to be cleaner than the rest
  const threshold = mode === "aggressive" ? 35 : 45;
  if (score <= threshold) return text;

  // Extra mutation specifically for the first paragraph
  const prompt = `You are doing a final cleanup pass on the OPENING PARAGRAPH only of a piece of writing. This paragraph needs to read as clearly human-written because it gets the most scrutiny.

TASK: Rewrite only to fix these specific issues:
- If two consecutive sentences start the same way, fix one opener.
- If all sentences are similar lengths, make one noticeably shorter or longer.
- If any formal connector appears (however/therefore/moreover), replace or remove it.
- If the paragraph ends with a tidy summary sentence, make it end more abruptly.

Do NOT restructure the paragraph. Do NOT change meaning. Fix only what is listed above.

OUTPUT: Only the rewritten paragraph. No preamble, no labels.

PARAGRAPH:
${firstPara}`;

  try {
    const improved = await callModel(prompt, {
      temperature: 0.85,
      topP: 0.92,
      topK: 35,
      frequencyPenalty: 0.25,
    });
    if (improved.trim().length > 0) {
      paragraphs[0] = improved.trim();
    }
  } catch {
    // If the extra call fails, keep the original first paragraph
  }

  return paragraphs.join("\n\n");
}

// ─── Anti-Pattern Destruction ────────────────────────────────────────────────
// Deterministic post-processing pass — no LLM call.
// Catches AI-signpost paragraph openers that models re-introduce despite
// prompt-level suppression. Each pattern maps to a pool of replacements;
// we cycle through them to avoid creating new repetitions.

const ANTI_PATTERNS: { regex: RegExp; replacements: string[] }[] = [
  { regex: /^Of course,?\s*/im,         replacements: ["", "Sure — ", "Look, ", "Right — "] },
  { regex: /^Now,?\s*/im,               replacements: ["", "So ", "At this point, ", "Then again, "] },
  { regex: /^Here'?s the thing:?\s*/im,  replacements: ["", "The thing is, ", "What matters here: ", "Put simply, "] },
  { regex: /^In conclusion,?\s*/im,      replacements: ["", "All told, ", "So — ", "Ultimately, "] },
  { regex: /^Moving toward\s*/im,        replacements: ["", "Shifting to ", "On to ", ""] },
  { regex: /^Nature,? too,?\s*/im,       replacements: ["Nature ", "The natural side ", "Even nature "] },
  { regex: /^No exploration of\s*/im,    replacements: ["You can't discuss ", "Any look at ", "It's hard to skip "] },
  { regex: /^Further south,?\s*/im,      replacements: ["To the south, ", "South of there, ", "Head south and "] },
  { regex: /^What's more,?\s*/im,        replacements: ["", "On top of that, ", "And ", "Plus, "] },
  { regex: /^It is worth noting\s*/im,   replacements: ["", "Note that ", "Worth flagging: ", ""] },
  { regex: /^Importantly,?\s*/im,        replacements: ["", "A key point: ", "What matters — ", ""] },
  { regex: /^Interestingly,?\s*/im,      replacements: ["", "One thing that stands out — ", "What's notable: ", ""] },
];

function antiPatternPass(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  let replacementIndex = 0;

  const cleaned = paragraphs.map((para) => {
    let result = para;
    for (const { regex, replacements } of ANTI_PATTERNS) {
      if (regex.test(result)) {
        const pick = replacements[replacementIndex % replacements.length];
        result = result.replace(regex, pick);
        replacementIndex++;
        break; // one replacement per paragraph — don't stack
      }
    }
    return result;
  });

  return cleaned.join("\n\n");
}

// ─── Filler Distribution Control ─────────────────────────────────────────────
// Deterministic pass — no LLM call.
// Prevents conversational paragraph openers from appearing too uniformly.
// If >30% of paragraphs start with casual filler, strip excess openers.

const CASUAL_OPENERS = [
  /^(You know,?\s*)/i,
  /^(Take\s)/i,
  /^(And\s)/i,
  /^(But\s)/i,
  /^(Look,?\s*)/i,
  /^(See,?\s*)/i,
  /^(So,?\s+)/i,
  /^(Sure,?\s*)/i,
  /^(Right,?\s*)/i,
  /^(Well,?\s*)/i,
  /^(Honestly,?\s*)/i,
  /^(Listen,?\s*)/i,
  /^(Thing is,?\s*)/i,
  /^(I mean,?\s*)/i,
];

function fillerDistributionPass(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  if (paragraphs.length < 3) return text;

  // Identify which paragraphs have casual openers
  const hasOpener = paragraphs.map((para) =>
    CASUAL_OPENERS.some((re) => re.test(para))
  );
  const openerCount = hasOpener.filter(Boolean).length;
  const ratio = openerCount / paragraphs.length;

  // Only act if casual openers exceed 30% of paragraphs
  if (ratio <= 0.3) return text;

  // Target: reduce to ~25% by stripping openers from mid-text paragraphs
  const targetKeep = Math.floor(paragraphs.length * 0.25);
  let kept = 0;

  const result = paragraphs.map((para, idx) => {
    if (!hasOpener[idx]) return para;

    // Always keep first and last paragraph openers if they have them
    if (idx === 0 || idx === paragraphs.length - 1) {
      kept++;
      return para;
    }

    // Keep up to targetKeep openers, strip the rest
    if (kept < targetKeep) {
      kept++;
      return para;
    }

    // Strip the opener — capitalise the remaining text
    let stripped = para;
    for (const re of CASUAL_OPENERS) {
      const match = stripped.match(re);
      if (match) {
        stripped = stripped.slice(match[0].length);
        if (stripped.length > 0) {
          stripped = stripped.charAt(0).toUpperCase() + stripped.slice(1);
        }
        break;
      }
    }
    return stripped;
  });

  return result.join("\n\n");
}

// ─── Sentence Asymmetry Injection ────────────────────────────────────────────
// Deterministic pass — no LLM call.
// Ensures each paragraph with 3+ sentences contains at least one noticeably
// shorter/plainer sentence. Prevents the "every sentence equally polished"
// AI fingerprint.
//
// Strategy:
//   1. Split paragraph into sentences.
//   2. If all sentences are within ±30% of the mean word count, the paragraph
//      is too uniform — pick one mid-paragraph sentence and shorten it.
//   3. Shortening: strip leading adverbial phrases, remove trailing qualifiers,
//      or truncate to the first clause.

// Trim common leading filler from a sentence to make it more direct.
const LEADING_FILLER = [
  /^(However|Moreover|Furthermore|Additionally|Consequently|Nevertheless|Indeed|Notably|Ultimately|Essentially|Fundamentally),?\s+/i,
  /^(It is|It's) (important|worth noting|clear|evident|crucial|essential) (to note |that )?/i,
  /^(In fact|As a result|For this reason|On the other hand|At the same time|In other words),?\s+/i,
  /^(What this means is|What's interesting is|The key thing is|The reality is|The point is),?\s+/i,
];

// Trim trailing qualifiers that add polish without substance.
const TRAILING_QUALIFIERS = [
  /,?\s+(which (is|makes|has been|remains) (particularly|especially|truly|quite|rather|deeply) (important|significant|notable|relevant|interesting))\.$/i,
  /,?\s+(and this (is|remains|has been) (particularly|especially|quite) (true|evident|clear|notable))\.$/i,
];

function shortenSentence(sentence: string): string {
  let result = sentence;

  // Strip one leading filler pattern
  for (const pattern of LEADING_FILLER) {
    const stripped = result.replace(pattern, "");
    if (stripped !== result && stripped.length > 10) {
      // Capitalise the new start
      result = stripped.charAt(0).toUpperCase() + stripped.slice(1);
      break;
    }
  }

  // Strip one trailing qualifier
  for (const pattern of TRAILING_QUALIFIERS) {
    const stripped = result.replace(pattern, ".");
    if (stripped !== result) {
      result = stripped;
      break;
    }
  }

  // If still long (>25 words), try truncating to first clause
  const words = result.split(/\s+/);
  if (words.length > 25) {
    const commaIdx = result.indexOf(",", 20);
    if (commaIdx > 0 && commaIdx < result.length - 15) {
      result = result.slice(0, commaIdx) + ".";
    }
  }

  return result;
}

function sentenceAsymmetryPass(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const processed = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 3) return para; // too short to need asymmetry

    const lengths = sentences.map((s) => s.split(/\s+/).length);
    const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;

    // Check if all sentences are within ±30% of mean (too uniform)
    const isUniform = lengths.every(
      (l) => l >= avg * 0.7 && l <= avg * 1.3
    );

    if (!isUniform) return para; // already has natural asymmetry

    // Pick a mid-paragraph sentence to flatten (not first or last)
    const targetIdx = Math.floor(sentences.length / 2);
    const shortened = shortenSentence(sentences[targetIdx]);
    sentences[targetIdx] = shortened;

    return sentences.join(" ");
  });

  return processed.join("\n\n");
}

// ─── Rhetorical Fluency Suppression ──────────────────────────────────────────
// Deterministic pass — no LLM call.
// Replaces polished-but-generic rhetorical phrases with grounded alternatives.
// Also flattens one sentence per paragraph if all sentences contain flourish.

const RHETORICAL_REPLACEMENTS: [RegExp, string][] = [
  // Phase 10 originals
  [/\btruly incredible\b/gi, "notable"],
  [/\bdeeply layered\b/gi, "layered"],
  [/\butterly rooted\b/gi, "rooted"],
  [/\bexact same moment\b/gi, "same moment"],
  [/\bbreathtaking\b/gi, "striking"],
  [/\bprofound(?:ly)?\b/gi, "real"],
  [/\bremarkable\b/gi, "worth noting"],
  [/\btruly transformative\b/gi, "significant"],
  [/\bdeeply rooted\b/gi, "rooted"],
  [/\bprofoundly important\b/gi, "important"],
  [/\bimmensely powerful\b/gi, "powerful"],
  [/\bfundamentally reshape\b/gi, "reshape"],
  [/\bparadigm shift\b/gi, "change"],
  [/\bgroundbreaking\b/gi, "new"],
  [/\bpivotal\b/gi, "key"],
  [/\btransformative\b/gi, "significant"],
  [/\bextraordinary\b/gi, "unusual"],
  [/\bseamlessly\b/gi, "smoothly"],
  [/\bintricate\b/gi, "detailed"],
  [/\bmeticulously\b/gi, "carefully"],
  [/\bundeniably\b/gi, "clearly"],
  [/\bindispensable\b/gi, "necessary"],
  [/\bunparalleled\b/gi, "unusual"],
  [/\boverarchingly\b/gi, "broadly"],
  [/\binextricably\b/gi, "closely"],
  // Phase 14 — vocabulary restraint (AI "upgrade" words → plain)
  [/\bgenuinely\b/gi, "really"],
  [/\butterly\b/gi, "completely"],
  [/\bmassive\b/gi, "large"],
  [/\bgargantuan\b/gi, "very large"],
  [/\bstartling\b/gi, "surprising"],
  [/\bpalpable\b/gi, "clear"],
  [/\bincredibly\b/gi, "very"],
  [/\bphenomenal\b/gi, "strong"],
  [/\bstaggering\b/gi, "large"],
  [/\bimmense\b/gi, "big"],
  [/\bprofuse\b/gi, "heavy"],
  [/\bvibrant\b/gi, "lively"],
  [/\brobust\b/gi, "strong"],
  [/\bcompelling\b/gi, "strong"],
  [/\boverwhelmingly\b/gi, "mostly"],
  [/\bundoubtedly\b/gi, "likely"],
  [/\bfascinating\b/gi, "interesting"],
  [/\bexceptionally\b/gi, "very"],
  [/\bresonates?\b/gi, "connects"],
  [/\bsignificantly\b/gi, "notably"],
  [/\bmetamorphos[ei]s\b/gi, "change"],
  [/\bcatalys[te]\b/gi, "driver"],
  [/\bparadigm\b/gi, "model"],
  [/\bholistic\b/gi, "broad"],
  [/\bsynergy\b/gi, "overlap"],
  [/\bleverage\b/gi, "use"],
  [/\binnovative\b/gi, "new"],
  [/\bcutting[- ]edge\b/gi, "modern"],
  [/\bstate[- ]of[- ]the[- ]art\b/gi, "current"],
];

// Words that mark a sentence as rhetorically "flourished"
const FLOURISH_MARKERS = /\b(profound|remarkable|extraordinary|transformative|breathtaking|pivotal|unprecedented|paradigm|groundbreaking|indispensable|unparalleled|meticulously|seamlessly|intricate|undeniably|genuinely|utterly|gargantuan|palpable|phenomenal|staggering|compelling|fascinating|exceptionally)\b/i;

function rhetoricalSuppressionPass(text: string): string {
  // Phase A: Direct phrase replacement
  let result = text;
  for (const [pattern, replacement] of RHETORICAL_REPLACEMENTS) {
    result = result.replace(pattern, replacement);
  }

  // Phase B: Ensure at least one plain sentence per paragraph
  const paragraphs = result.split(/\n\s*\n/);
  const processed = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 3) return para;

    // Check if every sentence still has a flourish marker
    const allFlourished = sentences.every((s) => FLOURISH_MARKERS.test(s));
    if (!allFlourished) return para;

    // Flatten the shortest sentence to be maximally plain
    const lengths = sentences.map((s) => s.split(/\s+/).length);
    const minIdx = lengths.indexOf(Math.min(...lengths));
    // Remove any remaining flourish words from that sentence
    sentences[minIdx] = sentences[minIdx].replace(FLOURISH_MARKERS, (match) => {
      const plainMap: Record<string, string> = {
        profound: "real", remarkable: "clear", extraordinary: "unusual",
        transformative: "big", breathtaking: "striking", pivotal: "key",
        unprecedented: "new", paradigm: "model", groundbreaking: "new",
        indispensable: "needed", unparalleled: "rare", meticulously: "carefully",
        seamlessly: "smoothly", intricate: "detailed", undeniably: "clearly",
      };
      return plainMap[match.toLowerCase()] ?? match;
    });

    return sentences.join(" ");
  });

  return processed.join("\n\n");
}

// ─── Human Writing Trait Engine ──────────────────────────────────────────────
// Controlled injection of 7 specific human writing traits.
// Each trait is measurable and applied deterministically — not random.
// Operates on the full text after all other passes.

// Trait 1: Non-uniform paragraph endings.
// AI ends every paragraph with a polished wrap-up sentence.
// Humans sometimes end flat — mid-thought, no bow-tie.
function flattenParagraphEndings(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  if (paragraphs.length < 3) return text;

  // Flatten ~30% of paragraph endings (every 3rd paragraph)
  const result = paragraphs.map((para, idx) => {
    if (idx === 0 || idx === paragraphs.length - 1) return para; // keep first and last
    if (idx % 3 !== 0) return para;

    const sentences = getSentences(para);
    if (sentences.length < 2) return para;

    const lastSentence = sentences[sentences.length - 1];
    // Remove trailing summarisation qualifiers
    const flatLast = lastSentence
      .replace(/,?\s+which (ultimately|essentially|fundamentally|clearly) .+\.$/i, ".")
      .replace(/,?\s+making (it|this|them) .+\.$/i, ".")
      .replace(/,?\s+and (this|that) (is|remains|has been) .+\.$/i, ".");

    if (flatLast !== lastSentence) {
      sentences[sentences.length - 1] = flatLast;
      return sentences.join(" ");
    }

    return para;
  });

  return result.join("\n\n");
}

// Trait 2: Selective plainness — ensure at least one sentence per paragraph
// is deliberately plain (under 15 words, no adjective clusters).
function ensureSelectivePlainness(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const result = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 3) return para;

    // Check if any sentence is already short and plain (< 12 words)
    const hasPlain = sentences.some((s) => s.split(/\s+/).length < 12);
    if (hasPlain) return para;

    // Find the shortest sentence and shorten it further
    const lengths = sentences.map((s) => s.split(/\s+/).length);
    const minIdx = lengths.indexOf(Math.min(...lengths));

    // Truncate to first clause boundary if possible
    const sentence = sentences[minIdx];
    const commaPos = sentence.indexOf(",", 10);
    if (commaPos > 0 && commaPos < sentence.length - 10) {
      sentences[minIdx] = sentence.slice(0, commaPos) + ".";
    }

    return sentences.join(" ");
  });

  return result.join("\n\n");
}

// Trait 3: Natural lexical recurrence.
// AI avoids repeating key words (synonym substitution). Humans repeat important
// words for emphasis. Find the most-used content word and allow it in 2 extra places.
// (This trait is passive — it prevents the synonym-substitution instinct from
// the LLM passes. We implement it by NOT replacing repeated key terms.)
// Already partially handled by topK=40 in semantic pass. No extra code needed.

// Trait 4: Controlled unfinished cadence.
// Some sentences should feel like they end a beat early.
// "That part becomes obvious quickly." instead of
// "That part becomes obvious quickly when you consider the broader context."
const TRAILING_EXTENSIONS = [
  /,?\s+when (you|we|one) consider[s]? .+\.$/i,
  /,?\s+especially (when|in|if|given|considering) .+\.$/i,
  /,?\s+particularly (in|when|for|given) .+\.$/i,
  /,?\s+which (helps|allows|enables|makes|ensures) .+\.$/i,
];

function controlledCadenceTrim(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  let trimCount = 0;

  const result = paragraphs.map((para) => {
    if (trimCount >= 3) return para; // max 3 trims per text

    const sentences = getSentences(para);
    if (sentences.length < 2) return para;

    let modified = false;
    const processed = sentences.map((sentence) => {
      if (modified || trimCount >= 3) return sentence;
      if (sentence.split(/\s+/).length < 18) return sentence; // already short

      for (const pattern of TRAILING_EXTENSIONS) {
        if (pattern.test(sentence)) {
          const trimmed = sentence.replace(pattern, ".");
          if (trimmed.split(/\s+/).length >= 8) { // don't over-trim
            trimCount++;
            modified = true;
            return trimmed;
          }
        }
      }
      return sentence;
    });

    return processed.join(" ");
  });

  return result.join("\n\n");
}

// Master trait engine — applies traits in sequence.
function humanTraitEngine(text: string): string {
  let result = text;
  result = flattenParagraphEndings(result);
  result = ensureSelectivePlainness(result);
  result = controlledCadenceTrim(result);
  return result;
}

// ─── Detector-Weight Implementation ──────────────────────────────────────────
// Implements specific findings from GPTZero/Originality.ai/QuillBot/ZeroGPT
// detector analysis. Each sub-pass targets a measurable signal.

// A/B: Opening sentence confidence reduction.
// Detectors assign highest weight to sentence 1. Flatten it.
function flattenOpeningSentence(text: string): string {
  const firstBreak = text.search(/(?<=[.!?])\s+/);
  if (firstBreak < 0) return text;

  let firstSentence = text.slice(0, firstBreak + 1);
  const rest = text.slice(firstBreak + 1);

  // Strip leading casual filler from first sentence
  firstSentence = firstSentence
    .replace(/^(Look,?\s*|So,?\s+|Here'?s the thing:?\s*|You know,?\s*|Well,?\s*)/i, "")
    .trim();

  // Strip trailing flourish from first sentence
  firstSentence = firstSentence
    .replace(/,?\s+which (is|makes|remains) .+\.$/i, ".")
    .replace(/\s+—\s+.+\.$/i, ".");

  // Ensure starts with uppercase
  if (firstSentence.length > 0) {
    firstSentence = firstSentence.charAt(0).toUpperCase() + firstSentence.slice(1);
  }

  return firstSentence + " " + rest.trimStart();
}

// C: Break consecutive upgraded phrases.
// If two consecutive sentences both contain flourish markers, the second
// gets its flourish word replaced with a plain equivalent.
function breakConsecutiveFlourish(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const result = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;

    for (let i = 1; i < sentences.length; i++) {
      const prevHasFlourish = FLOURISH_MARKERS.test(sentences[i - 1]);
      const currHasFlourish = FLOURISH_MARKERS.test(sentences[i]);

      if (prevHasFlourish && currHasFlourish) {
        // Flatten the current sentence's flourish
        sentences[i] = sentences[i].replace(FLOURISH_MARKERS, (match) => {
          const plainMap: Record<string, string> = {
            profound: "real", remarkable: "clear", extraordinary: "unusual",
            transformative: "big", breathtaking: "striking", pivotal: "key",
            unprecedented: "new", paradigm: "model", groundbreaking: "new",
            indispensable: "needed", unparalleled: "rare", meticulously: "carefully",
            seamlessly: "smoothly", intricate: "detailed", undeniably: "clearly",
            genuinely: "really", utterly: "completely", gargantuan: "very large",
            palpable: "clear", phenomenal: "strong", staggering: "large",
            compelling: "solid", fascinating: "interesting", exceptionally: "very",
          };
          return plainMap[match.toLowerCase()] ?? match;
        });
      }
    }

    return sentences.join(" ");
  });

  return result.join("\n\n");
}

// D: Vary paragraph energy deliberately.
// If consecutive paragraphs have similar "energy" (word count within 20%),
// compress one by trimming its longest sentence at first clause boundary.
function varyParagraphEnergy(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  if (paragraphs.length < 3) return text;

  const wordCounts = paragraphs.map((p) => p.split(/\s+/).length);

  for (let i = 1; i < paragraphs.length - 1; i++) {
    const prev = wordCounts[i - 1];
    const curr = wordCounts[i];

    // Check if energy is too similar to neighbor
    const ratio = curr / Math.max(prev, 1);
    if (ratio < 0.8 || ratio > 1.2) continue; // already varied

    // Compress this paragraph by shortening its longest sentence
    const sentences = getSentences(paragraphs[i]);
    if (sentences.length < 2) continue;

    const lengths = sentences.map((s) => s.split(/\s+/).length);
    const maxIdx = lengths.indexOf(Math.max(...lengths));

    if (lengths[maxIdx] > 15) {
      const sentence = sentences[maxIdx];
      const commaPos = sentence.indexOf(",", 12);
      if (commaPos > 0 && commaPos < sentence.length - 10) {
        sentences[maxIdx] = sentence.slice(0, commaPos) + ".";
        paragraphs[i] = sentences.join(" ");
        wordCounts[i] = paragraphs[i].split(/\s+/).length;
      }
    }
  }

  return paragraphs.join("\n\n");
}

function detectorWeightPass(text: string): string {
  let result = text;
  result = flattenOpeningSentence(result);
  result = breakConsecutiveFlourish(result);
  result = varyParagraphEnergy(result);
  return result;
}

// ─── Low-Mutation Islands ────────────────────────────────────────────────────
// After all LLM passes, find 1-2 sentences from the original input that
// closely match sentences in the output (high word overlap) and restore
// them closer to their original form. Creates "islands" of minimal mutation
// within the rewritten text, breaking the full-rewrite fingerprint.

function computeWordOverlap(a: string, b: string): number {
  const wordsA = new Set(a.toLowerCase().replace(/[^a-z\s]/g, "").split(/\s+/));
  const wordsB = new Set(b.toLowerCase().replace(/[^a-z\s]/g, "").split(/\s+/));
  let overlap = 0;
  for (const w of wordsA) {
    if (wordsB.has(w)) overlap++;
  }
  const union = new Set([...wordsA, ...wordsB]).size;
  return union === 0 ? 0 : overlap / union; // Jaccard similarity
}

function lowMutationIslands(output: string, originalInput: string): string {
  const outputSentences = output
    .split(/(?<=[.!?])\s+/)
    .filter((s) => s.trim().length > 10);
  const inputSentences = originalInput
    .split(/(?<=[.!?])\s+/)
    .filter((s) => s.trim().length > 10);

  if (outputSentences.length < 5 || inputSentences.length < 3) return output;

  // Find output sentences that have highest overlap with an original sentence
  const candidates: { outIdx: number; inIdx: number; overlap: number }[] = [];

  for (let oi = 0; oi < outputSentences.length; oi++) {
    for (let ii = 0; ii < inputSentences.length; ii++) {
      const overlap = computeWordOverlap(outputSentences[oi], inputSentences[ii]);
      if (overlap > 0.5) {
        candidates.push({ outIdx: oi, inIdx: ii, overlap });
      }
    }
  }

  // Sort by overlap descending, pick up to 2 non-adjacent islands
  candidates.sort((a, b) => b.overlap - a.overlap);

  const restored = new Set<number>();
  for (const c of candidates) {
    if (restored.size >= 2) break;
    // Don't restore first or last sentence (those are handled by other passes)
    if (c.outIdx === 0 || c.outIdx === outputSentences.length - 1) continue;
    // Don't restore adjacent to already restored
    const hasAdjacentRestore = [...restored].some(
      (r) => Math.abs(r - c.outIdx) <= 1
    );
    if (hasAdjacentRestore) continue;

    // Restore the original input sentence into this position
    outputSentences[c.outIdx] = inputSentences[c.inIdx];
    restored.add(c.outIdx);
  }

  return outputSentences.join(" ");
}

// ─── Length Discipline ───────────────────────────────────────────────────────
// If the output exceeds 110% of input word count, trim at the last complete
// sentence boundary that fits within the budget. Never cuts mid-sentence.

function enforceLengthDiscipline(
  output: string,
  inputWordCount: number
): string {
  const maxWords = Math.ceil(inputWordCount * 1.10);
  const outputWords = output.split(/\s+/).length;

  if (outputWords <= maxWords) return output; // within budget

  // Split into sentences and accumulate until budget hit
  const allSentences = output
    .split(/(?<=[.!?])\s+/)
    .filter((s) => s.trim().length > 0);

  let wordsSoFar = 0;
  const kept: string[] = [];

  for (const sentence of allSentences) {
    const sentenceWords = sentence.split(/\s+/).length;
    if (wordsSoFar + sentenceWords > maxWords && kept.length > 0) break;
    kept.push(sentence);
    wordsSoFar += sentenceWords;
  }

  return kept.join(" ");
}

// ─── Public API ─────────────────────────────────────────────────────────────

export async function humanize(
  inputText: string,
  mode: HumanizeMode,
  wordLimit: number
): Promise<string> {
  const truncated = truncateToWordLimit(inputText, wordLimit);
  const inputWordCount = truncated.split(/\s+/).length;

  // Split into variable-size chunks for natural local inconsistency
  const chunks = splitIntoVariableChunks(truncated);

  // Pass 1: Structural rewrite — parallel per chunk
  const structural = await structuralPass(chunks, mode);

  // Pass 2: Semantic naturalness — parallel per chunk (skipped for light)
  const semantic = await semanticPass(structural, mode);

  // Merge chunks back into full text
  const merged = semantic.join("\n\n");

  // Pass 3: Targeted mutation on full merged text (gated by score)
  const mutated = await mutationPass(merged, mode);

  // Pass 3b: First-paragraph hardening (extra mutation if opener still scores high)
  const hardened = await firstParagraphHardening(mutated, mode);

  // Pass 4: Deterministic anti-pattern cleanup (no LLM call)
  const cleaned = antiPatternPass(hardened);

  // Pass 4b: Filler distribution control (no LLM call)
  const fillerControlled = fillerDistributionPass(cleaned);

  // Pass 5: Sentence asymmetry injection (no LLM call)
  const asymmetric = sentenceAsymmetryPass(fillerControlled);

  // Pass 6: Rhetorical fluency suppression (no LLM call)
  const suppressed = rhetoricalSuppressionPass(asymmetric);

  // Pass 7: Human writing trait engine (no LLM call)
  const humanized = humanTraitEngine(suppressed);

  // Pass 8: Detector-weight implementation (no LLM call)
  const detectorHardened = detectorWeightPass(humanized);

  // Pass 9: Low-mutation islands (no LLM call) — restore 1-2 original sentences
  const islanded = lowMutationIslands(detectorHardened, truncated);

  // Pass 10: Length discipline — enforce 90%–110% of input word count
  return enforceLengthDiscipline(islanded, inputWordCount);
}
