import { GoogleGenerativeAI } from "@google/generative-ai";
import { detectAI } from "@/lib/detector";
import {
  getStructuralPrompt,
  getSemanticPrompt,
  getMutationPrompt,
  type HumanizeMode,
  type SourceRegister as PromptSourceRegister,
} from "@/prompts/pipeline";

export type { HumanizeMode };

// ─── Source Register Classifier ─────────────────────────────────────────────
// Classifies the INPUT text register before any transformation.
// This classification acts as a hard ceiling on tone drift throughout the
// pipeline — late passes must not inject markers outside the source register.

export type SourceRegister = "academic" | "formal" | "neutral" | "informal";

const ACADEMIC_MARKERS = /\b(furthermore|moreover|consequently|notwithstanding|henceforth|thereby|wherein|thus|hence|herein|aforementioned|et al|i\.e\.|e\.g\.)\b/i;
const FORMAL_MARKERS = /\b(therefore|however|nevertheless|regarding|pertaining|respectively|accordingly|subsequent|preceding)\b/i;
const INFORMAL_MARKERS = /\b(gonna|wanna|gotta|kinda|sorta|yeah|nah|okay|ok|hey|yep|nope|stuff|thing is|you know|I mean|right\?|honestly)\b/i;
const CONTRACTION_RE = /\b(I'm|you're|we're|they're|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't|it's|that's|there's|here's|what's|who's|let's|I've|you've|we've|they've|I'll|you'll|we'll|they'll|I'd|you'd|we'd|they'd)\b/gi;

export function classifyRegister(text: string): SourceRegister {
  const words = text.split(/\s+/).length;
  const sentences = text.split(/(?<=[.!?])\s+/).length;
  const avgSentLen = words / Math.max(sentences, 1);

  // Count markers
  const academicHits = (text.match(ACADEMIC_MARKERS) || []).length;
  const formalHits = (text.match(FORMAL_MARKERS) || []).length;
  const informalHits = (text.match(INFORMAL_MARKERS) || []).length;
  const contractionCount = (text.match(CONTRACTION_RE) || []).length;
  const contractionDensity = contractionCount / Math.max(words, 1) * 100;

  // High contraction density + informal markers → informal
  if (contractionDensity > 2.5 || informalHits >= 3) return "informal";
  if (informalHits >= 2 && contractionDensity > 1.5) return "informal";

  // Academic markers or very long avg sentence + formal vocabulary
  if (academicHits >= 2) return "academic";
  if (academicHits >= 1 && avgSentLen > 22) return "academic";
  if (formalHits >= 3 && avgSentLen > 20) return "academic";

  // Formal markers + no contractions + moderate sentence length
  if (formalHits >= 2 && contractionDensity < 1) return "formal";
  if (avgSentLen > 20 && contractionDensity < 0.5) return "formal";

  // Neutral: moderate everything
  return "neutral";
}

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
    "https://router.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2";

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
  mode: HumanizeMode,
  register?: SourceRegister
): Promise<string[]> {
  const settings = STRUCTURAL_SETTINGS[mode];
  return Promise.all(
    chunks.map((chunk, i) => {
      const { annotated } = annotateChunkWithClasses(chunk);
      return callModel(
        getStructuralPrompt(chunk, mode, i, annotated, register),
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
  mode: HumanizeMode,
  register?: SourceRegister
): Promise<string[]> {
  if (mode === "light") return chunks;
  const settings = SEMANTIC_SETTINGS[mode];
  return Promise.all(
    chunks.map((chunk, i) =>
      callModel(getSemanticPrompt(chunk, mode, i, register as PromptSourceRegister), settings)
        .then((out) => validateChunkOutput(out, chunk))
        .catch(() => chunk) // fall back to structural output on any failure
    )
  );
}

// Pass 3 — Selective mutation on the full merged text.
// Runs once on the assembled output, not per chunk — short chunks give
// noisy detector readings. Gated by score threshold; skipped if already good.
// Exception: short text (< 120 words) always runs mutation — it lacks the
// paragraph variance that makes longer text pass ZeroGPT's perplexity check.
async function mutationPass(
  text: string,
  mode: HumanizeMode
): Promise<string> {
  if (mode === "light") return text;

  const wordCount = text.split(/\s+/).length;
  const isShort = wordCount < 120;

  const threshold = mode === "aggressive" ? 42 : 52;
  const { score } = detectAI(text);

  // Short text: always mutate regardless of score (ZeroGPT is perplexity-based
  // and won't be fooled by surface-level passes alone on short text)
  if (!isShort && score <= threshold) return text;

  return callModel(getMutationPrompt(text), MUTATION_SETTINGS);
}

// ─── Short-Text Perplexity Hardening ────────────────────────────────────────
// For text under 120 words, ZeroGPT's perplexity model sees through generic
// LLM rewrites because there's no paragraph-level variance to hide behind.
// This dedicated pass forces unpredictability into short text by asking the
// model to make specific structural breaks.

async function shortTextPerplexityHardening(
  text: string,
  mode: HumanizeMode,
  register: SourceRegister
): Promise<string> {
  if (mode === "light") return text;
  const wordCount = text.split(/\s+/).length;
  if (wordCount >= 120) return text; // only for short text

  const isFormal = register === "academic" || register === "formal";

  // For academic/formal: structural variation without casual drift
  // For neutral/informal: allow contractions and everyday replacements
  const contractionRule = isFormal
    ? "2. Do NOT add contractions — preserve the formal register."
    : "2. Add one contraction that isn't already there (it's, that's, you'll, we've, isn't, etc.)";

  const wordRule = isFormal
    ? "4. Replace one overly complex word with a simpler but still formal equivalent (e.g., utilize→use, demonstrate→show)"
    : "4. Change one formal word to an everyday word (utilize→use, demonstrate→show, significant→real, etc.)";

  const punctuationRule = isFormal
    ? "5. If every sentence ends with a period, vary one with a colon or semicolon"
    : "5. If every sentence ends with a period, convert one to end with a dash or question";

  const prompt = `You are editing a SHORT piece of text (under 120 words).
The goal is to make it read as completely human-written — specifically to pass perplexity-based AI detectors.
IMPORTANT: Preserve the original tone and register. ${isFormal ? "This text is formal/academic — keep it formal." : ""}

REQUIRED changes — apply ALL of these:
1. Break one sentence into two shorter ones (split at a natural point)
${contractionRule}
3. Make one sentence start with a word that's unexpected given the context (not "The", "This", "It", "However")
${wordRule}
${punctuationRule}

OUTPUT: Only the revised text. No labels, no commentary, no preamble. Same meaning, same topic, same register.

TEXT:
${text}`;

  try {
    const result = await callModel(prompt, {
      temperature: 0.92,
      topP: 0.95,
      topK: 40,
      frequencyPenalty: 0.30,
    });
    if (result.trim().length > 20) return result.trim();
  } catch {
    // fall through to original
  }
  return text;
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

// Casual replacements forbidden for academic/formal register
const CASUAL_REPLACEMENT_RE = /^(Sure|Look|Right|So\b|Then again|All told|Plus|The thing is|One thing that stands out|What matters|Worth flagging)/i;

function antiPatternPass(text: string, register: SourceRegister): string {
  const paragraphs = text.split(/\n\s*\n/);
  let replacementIndex = 0;
  const isFormal = register === "academic" || register === "formal";

  const cleaned = paragraphs.map((para) => {
    let result = para;
    for (const { regex, replacements } of ANTI_PATTERNS) {
      if (regex.test(result)) {
        // For formal/academic: filter to only register-safe replacements
        const pool = isFormal
          ? replacements.filter((r) => r === "" || !CASUAL_REPLACEMENT_RE.test(r))
          : replacements;
        const safePool = pool.length > 0 ? pool : [""];
        const pick = safePool[replacementIndex % safePool.length];
        result = result.replace(regex, pick);
        replacementIndex++;
        break;
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

// ─── Deterministic Micro-Surgery ─────────────────────────────────────────────
// Phase 18: Code-based (no LLM) surgical pass that breaks model-family
// fingerprint after all generation passes. Three sub-passes:
//   A) If a paragraph is "too polished" (all sentences fluent, no short/plain
//      sentence), flatten one sentence by stripping its trailing clause.
//   B) If two upgraded adjectives appear within 40 words of each other, downgrade
//      the second one to a plain word.
//   C) If two consecutive sentences both contain rhetorical connectors, remove
//      the connector from the second sentence.

// Upgraded adjectives that AI sprinkles in — the "tell" is two appearing close together.
const UPGRADED_ADJECTIVES: { re: RegExp; plain: string }[] = [
  { re: /\bsignificant\b/i,   plain: "real" },
  { re: /\bsubstantial\b/i,   plain: "large" },
  { re: /\bnotable\b/i,       plain: "clear" },
  { re: /\bcritical\b/i,      plain: "important" },
  { re: /\bcomprehensive\b/i, plain: "full" },
  { re: /\bextensive\b/i,     plain: "wide" },
  { re: /\bsophisticated\b/i, plain: "complex" },
  { re: /\bdiverse\b/i,       plain: "varied" },
  { re: /\bdynamic\b/i,       plain: "active" },
  { re: /\bessential\b/i,     plain: "needed" },
  { re: /\bcrucial\b/i,       plain: "key" },
  { re: /\bvital\b/i,         plain: "key" },
  { re: /\bstriking\b/i,      plain: "clear" },
  { re: /\bcomplex\b/i,       plain: "detailed" },
  { re: /\beffective\b/i,     plain: "useful" },
];

// Rhetorical connectors that feel editorial when consecutive.
const RHETORICAL_CONNECTORS = /^(In particular|Specifically|More importantly|Notably|Significantly|What stands out|What matters|The key point|A key factor|Central to this|At the heart of|Fundamentally|Crucially|Critically),?\s+/i;

// Sub-pass A: Flatten one sentence in over-polished paragraphs.
// "Over-polished" = all sentences ≥15 words with no short/plain sentence.
function flattenOverPolishedParagraph(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  let flattened = 0;

  const result = paragraphs.map((para) => {
    if (flattened >= 2) return para; // max 2 flattenings per text

    const sentences = getSentences(para);
    if (sentences.length < 3) return para;

    const lengths = sentences.map((s) => s.split(/\s+/).length);
    const hasShort = lengths.some((l) => l < 13);
    if (hasShort) return para; // already has a short sentence — not over-polished

    // All sentences are 13+ words. Flatten a mid-paragraph sentence.
    const targetIdx = 1 + (flattened % (sentences.length - 2)); // avoid first/last
    const safeIdx = Math.min(targetIdx, sentences.length - 2);
    const sentence = sentences[safeIdx];

    // Strategy: truncate at first comma or semicolon after 8+ words
    const words = sentence.split(/\s+/);
    if (words.length < 16) return para; // not long enough to truncate safely

    let truncated = sentence;
    const semicolonPos = sentence.indexOf(";", 20);
    const commaPos = sentence.indexOf(",", 20);
    const cutPos = semicolonPos > 0 ? semicolonPos : commaPos;

    if (cutPos > 0 && cutPos < sentence.length - 15) {
      truncated = sentence.slice(0, cutPos).trimEnd() + ".";
    }

    if (truncated !== sentence && truncated.split(/\s+/).length >= 6) {
      sentences[safeIdx] = truncated;
      flattened++;
      return sentences.join(" ");
    }

    return para;
  });

  return result.join("\n\n");
}

// Sub-pass B: Adjacent upgraded adjective trimming.
// Scans the full text for two upgraded adjectives within a 40-word window.
// Downgrades the second occurrence to its plain equivalent.
function trimAdjacentUpgradedAdjectives(text: string): string {
  // Find all upgraded adjective positions
  const matches: { index: number; length: number; adjIdx: number }[] = [];
  for (let ai = 0; ai < UPGRADED_ADJECTIVES.length; ai++) {
    const { re } = UPGRADED_ADJECTIVES[ai];
    const globalRe = new RegExp(re.source, "gi");
    let m: RegExpExecArray | null;
    while ((m = globalRe.exec(text)) !== null) {
      matches.push({ index: m.index, length: m[0].length, adjIdx: ai });
    }
  }

  if (matches.length < 2) return text;
  matches.sort((a, b) => a.index - b.index);

  // Find pairs within 40-word windows and mark the second for downgrade
  const toReplace = new Set<number>(); // indices into matches[]
  for (let i = 0; i < matches.length - 1; i++) {
    if (toReplace.has(i)) continue;
    const span = text.slice(matches[i].index, matches[i + 1].index + matches[i + 1].length);
    const wordsBetween = span.split(/\s+/).length;
    if (wordsBetween <= 40) {
      toReplace.add(i + 1); // mark the second one
    }
  }

  if (toReplace.size === 0) return text;

  // Apply replacements from end to start to preserve indices
  const sortedReplacements = [...toReplace].sort((a, b) => matches[b].index - matches[a].index);
  let result = text;
  for (const mi of sortedReplacements) {
    const { index, length, adjIdx } = matches[mi];
    const original = result.slice(index, index + length);
    let plain = UPGRADED_ADJECTIVES[adjIdx].plain;
    // Preserve capitalisation
    if (original[0] === original[0].toUpperCase()) {
      plain = plain.charAt(0).toUpperCase() + plain.slice(1);
    }
    result = result.slice(0, index) + plain + result.slice(index + length);
  }

  return result;
}

// Sub-pass C: Consecutive rhetorical connector reduction.
// If two adjacent sentences both start with rhetorical connectors, strip
// the connector from the second sentence.
function reduceConsecutiveRhetoricalConnectors(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const result = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;

    for (let i = 1; i < sentences.length; i++) {
      const prevHas = RHETORICAL_CONNECTORS.test(sentences[i - 1]);
      const currHas = RHETORICAL_CONNECTORS.test(sentences[i]);

      if (prevHas && currHas) {
        // Strip the connector from the current sentence
        const stripped = sentences[i].replace(RHETORICAL_CONNECTORS, "");
        if (stripped.length > 10) {
          sentences[i] = stripped.charAt(0).toUpperCase() + stripped.slice(1);
        }
      }
    }

    return sentences.join(" ");
  });

  return result.join("\n\n");
}

// Master micro-surgery pass — applies all three sub-passes.
function deterministicMicroSurgery(text: string): string {
  let result = text;
  result = flattenOverPolishedParagraph(result);
  result = trimAdjacentUpgradedAdjectives(result);
  result = reduceConsecutiveRhetoricalConnectors(result);
  return result;
}

// ─── Multi-Detector Hardening ────────────────────────────────────────────────
// Phase 19: Targets statistical signals that GPTZero, ZeroGPT, QuillBot, and
// Originality.ai use but our internal detector doesn't fully capture.
// Three sub-passes — all deterministic, no LLM calls.

// Sub-pass A: N-gram frequency suppression.
// AI text reuses the same 2-word combos (bigrams) at much higher rates than
// human text. Find bigrams that appear 3+ times and replace excess occurrences
// with light paraphrases.
const OVERUSED_BIGRAMS: { bigram: string; replacements: string[] }[] = [
  { bigram: "would likely",    replacements: ["will probably", "could well", "might"] },
  { bigram: "tend to",         replacements: ["often", "usually", "frequently"] },
  { bigram: "in order",        replacements: ["to", "so as to", "for"] },
  { bigram: "as well",         replacements: ["too", "also", "on top of that"] },
  { bigram: "due to",          replacements: ["because of", "thanks to", "from"] },
  { bigram: "such as",         replacements: ["like", "including", "for example"] },
  { bigram: "it is",           replacements: ["it's", "this is", "that's"] },
  { bigram: "there is",        replacements: ["there's", "you'll find", "we see"] },
  { bigram: "there are",       replacements: ["there're", "you'll find", "we see"] },
  { bigram: "has been",        replacements: ["was", "has turned out", "became"] },
  { bigram: "can be",          replacements: ["is sometimes", "may be", "often is"] },
  { bigram: "one of",          replacements: ["among", "a top", "a major"] },
  { bigram: "need to",         replacements: ["must", "should", "have to"] },
  { bigram: "able to",         replacements: ["can", "capable of", "in a position to"] },
  { bigram: "important to",    replacements: ["worth", "key to", "useful to"] },
];

// Bigram replacements that inject casual tone (contractions, "you'll find", etc.)
const CASUAL_BIGRAM_REPLACEMENTS = new Set(["it's", "that's", "there's", "there're", "you'll find", "we see", "on top of that"]);

function ngramFrequencySuppression(text: string, register: SourceRegister): string {
  let result = text;
  const isFormal = register === "academic" || register === "formal";

  for (const { bigram, replacements: rawReplacements } of OVERUSED_BIGRAMS) {
    // Filter casual replacements for formal register
    const replacements = isFormal
      ? rawReplacements.filter((r) => !CASUAL_BIGRAM_REPLACEMENTS.has(r))
      : rawReplacements;
    if (replacements.length === 0) continue;
    const regex = new RegExp(`\\b${bigram}\\b`, "gi");
    const matches = [...result.matchAll(regex)];

    if (matches.length < 3) continue; // only suppress if 3+ occurrences

    // Keep first occurrence, replace alternating excess from the end
    let replaceCount = 0;
    const maxReplacements = Math.floor(matches.length / 2); // replace ~half

    // Process matches from end to preserve string indices
    for (let i = matches.length - 1; i >= 1 && replaceCount < maxReplacements; i -= 2) {
      const match = matches[i];
      if (match.index === undefined) continue;
      const replacement = replacements[replaceCount % replacements.length];
      result =
        result.slice(0, match.index) +
        replacement +
        result.slice(match.index + match[0].length);
      replaceCount++;
    }
  }

  return result;
}

// Sub-pass B: Predictable continuation breaker.
// AI frequently produces stock opening patterns with extremely high token
// continuation probability. Break them by rewriting the pattern.
const PREDICTABLE_PATTERNS: [RegExp, string][] = [
  [/\bOne of the most (important|significant|critical|notable)\b/gi, "A particularly $1"],
  [/\bIt is also worth\b/gi, "Also worth"],
  [/\bThis is particularly (true|evident|clear|notable)\b/gi, "That holds especially"],
  [/\bThis has led to\b/gi, "The result:"],
  [/\bThis means that\b/gi, "So"],
  [/\bThis is because\b/gi, "The reason:"],
  [/\bThere has been a growing\b/gi, "We've seen more"],
  [/\bIt is also important\b/gi, "Also key"],
  [/\bAs a result of this\b/gi, "Because of this"],
  [/\bIn recent years,?\s/gi, "Lately, "],
  [/\bAt the same time\b/gi, "Meanwhile"],
  [/\bOn the other hand\b/gi, "Then again"],
  [/\bFor this reason\b/gi, "That's why"],
  [/\bAs such\b/gi, "So"],
  [/\bGiven that\b/gi, "Since"],
  [/\bWith that said\b/gi, "Still"],
  [/\bWith this in mind\b/gi, "Keeping that in mind"],
  [/\bTo that end\b/gi, "For that"],
  [/\bBy and large\b/gi, "Mostly"],
  [/\bAll things considered\b/gi, "Overall"],
];

function predictableContinuationBreaker(text: string): string {
  let result = text;
  for (const [pattern, replacement] of PREDICTABLE_PATTERNS) {
    result = result.replace(pattern, replacement);
  }
  return result;
}

// Sub-pass C: Paragraph-level vocabulary monotony breaker.
// If a paragraph has very low type-token ratio (< 0.45), it signals AI
// repetitiveness. Find the most repeated non-stop content word and replace
// one occurrence with a pronoun or demonstrative.
const STOP_WORDS = new Set([
  "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "shall",
  "should", "may", "might", "must", "can", "could", "to", "of", "in",
  "for", "on", "with", "at", "by", "from", "as", "into", "through",
  "during", "before", "after", "and", "but", "or", "nor", "not", "so",
  "yet", "both", "either", "neither", "each", "every", "all", "any",
  "few", "more", "most", "other", "some", "such", "no", "only", "own",
  "same", "than", "too", "very", "just", "about", "also", "then",
  "that", "this", "these", "those", "it", "its", "they", "them", "their",
  "he", "she", "his", "her", "we", "our", "you", "your", "i", "my",
  "me", "which", "who", "whom", "what", "where", "when", "how", "if",
]);

function paragraphVocabularyBreaker(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const result = paragraphs.map((para) => {
    const words = para.toLowerCase().replace(/[^a-z\s]/g, " ").split(/\s+/).filter(w => w.length > 2);
    if (words.length < 20) return para; // too short to measure

    const unique = new Set(words).size;
    const ttr = unique / words.length;

    if (ttr >= 0.45) return para; // diversity is acceptable

    // Find most repeated content word
    const freq: Record<string, number> = {};
    for (const w of words) {
      if (STOP_WORDS.has(w)) continue;
      freq[w] = (freq[w] || 0) + 1;
    }

    const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]);
    if (sorted.length === 0 || sorted[0][1] < 3) return para; // no word repeats 3+ times

    const overusedWord = sorted[0][0];
    // Replace the second occurrence with "this" or "it" where grammatically safe
    const wordRegex = new RegExp(`\\b${overusedWord}\\b`, "gi");
    let occurrenceCount = 0;
    const fixed = para.replace(wordRegex, (match) => {
      occurrenceCount++;
      // Replace the 2nd occurrence only
      if (occurrenceCount === 2) return "this";
      return match;
    });

    return fixed;
  });

  return result.join("\n\n");
}

// ─── Perplexity Injector ─────────────────────────────────────────────────────
// ZeroGPT is heavily perplexity-based — it flags text where token sequences
// are too predictable under a language model. This pass swaps one common word
// per paragraph with a lower-frequency synonym from a curated pool, increasing
// per-token surprise without changing meaning.
// Register-gated: only formalSafe swaps are used for academic/formal text.

interface SynonymSwap {
  word: string;
  replacements: string[];
  formalSafe: boolean;
}

const PERPLEXITY_SWAP_POOL: SynonymSwap[] = [
  { word: "shows",     replacements: ["illustrates", "reveals", "indicates"],      formalSafe: true  },
  { word: "show",      replacements: ["illustrate", "reveal", "indicate"],          formalSafe: true  },
  { word: "shown",     replacements: ["illustrated", "demonstrated", "indicated"],  formalSafe: true  },
  { word: "uses",      replacements: ["employs", "applies", "utilizes"],            formalSafe: true  },
  { word: "use",       replacements: ["employ", "apply", "utilize"],                formalSafe: true  },
  { word: "many",      replacements: ["numerous", "various", "several"],            formalSafe: true  },
  { word: "because",   replacements: ["since", "given that", "as"],                formalSafe: true  },
  { word: "important", replacements: ["significant", "notable", "critical"],        formalSafe: true  },
  { word: "often",     replacements: ["frequently", "commonly", "regularly"],       formalSafe: true  },
  { word: "need",      replacements: ["require", "necessitate", "demand"],          formalSafe: true  },
  { word: "help",      replacements: ["assist", "facilitate", "support"],           formalSafe: true  },
  { word: "find",      replacements: ["identify", "observe", "discover"],           formalSafe: true  },
  { word: "change",    replacements: ["alter", "shift", "transform"],               formalSafe: true  },
  { word: "start",     replacements: ["begin", "initiate", "commence"],             formalSafe: true  },
  { word: "think",     replacements: ["consider", "argue", "suggest"],              formalSafe: true  },
  { word: "very",      replacements: ["considerably", "notably", "quite"],          formalSafe: true  },
  { word: "also",      replacements: ["likewise", "equally", "as well"],            formalSafe: true  },
  { word: "get",       replacements: ["obtain", "acquire", "gain"],                 formalSafe: false },
  { word: "good",      replacements: ["effective", "beneficial", "favorable"],      formalSafe: false },
  { word: "big",       replacements: ["substantial", "considerable", "significant"], formalSafe: false },
  { word: "small",     replacements: ["limited", "modest", "minimal"],              formalSafe: false },
];

function perplexityInjector(text: string, register: SourceRegister): string {
  const isFormal = register === "academic" || register === "formal";
  const pool = isFormal
    ? PERPLEXITY_SWAP_POOL.filter((s) => s.formalSafe)
    : PERPLEXITY_SWAP_POOL;

  const paragraphs = text.split(/\n\s*\n/);

  const result = paragraphs.map((para) => {
    // One swap per paragraph — find first match and apply once
    for (const swap of pool) {
      const re = new RegExp(`\\b${swap.word}\\b`, "i");
      if (!re.test(para)) continue;

      // Pick replacement using paragraph length as a stable hash for variety
      const idx = para.length % swap.replacements.length;
      const replacement = swap.replacements[idx];

      // Replace first occurrence only, preserving original casing
      return para.replace(re, (m) =>
        m[0] === m[0].toUpperCase() && m[0] !== m[0].toLowerCase()
          ? replacement[0].toUpperCase() + replacement.slice(1)
          : replacement
      );
    }
    return para;
  });

  return result.join("\n\n");
}

// Master multi-detector hardening pass.
function multiDetectorHardening(text: string, register: SourceRegister): string {
  let result = text;
  result = ngramFrequencySuppression(result, register);
  result = predictableContinuationBreaker(result);
  result = paragraphVocabularyBreaker(result);
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

// ─── Stylometric Correction Layer ────────────────────────────────────────────
// Phase 20: Hybrid post-processing that analyzes final output sentence-by-
// sentence and applies selective deterministic corrections ONLY where local
// polish score indicates excessive smoothness. This creates a text that is
// partly model-generated and partly rule-shaped — breaking model-family
// signatures that survive all earlier passes.
//
// Five correction types, each gated by a per-sentence or per-paragraph
// polish score so they never fire everywhere at once.

// Per-sentence polish score (0-100). Higher = more polished/AI-like.
function sentencePolishScore(sentence: string): number {
  let score = 0;
  const words = sentence.split(/\s+/);
  const wordCount = words.length;

  // Length in the AI sweet spot (18-28 words) → polished
  if (wordCount >= 18 && wordCount <= 28) score += 25;
  else if (wordCount >= 14 && wordCount <= 32) score += 10;

  // Contains flourish/upgrade words
  if (FLOURISH_MARKERS.test(sentence)) score += 20;

  // Contains formal connectors
  if (/\b(however|therefore|moreover|furthermore|additionally|consequently|nevertheless)\b/i.test(sentence)) score += 20;

  // Balanced comma structure (2+ commas, roughly evenly spaced)
  const commas = (sentence.match(/,/g) || []).length;
  if (commas >= 2) {
    const parts = sentence.split(",").map(p => p.trim().split(/\s+/).length);
    const avg = parts.reduce((a, b) => a + b, 0) / parts.length;
    const allSimilar = parts.every(p => p >= avg * 0.6 && p <= avg * 1.4);
    if (allSimilar) score += 15;
  }

  // Starts with a common AI opener pattern
  if (/^(This|These|The|It|They|While|Although|Given)\b/.test(sentence)) score += 10;

  // Ends with a tidy conclusion phrase
  if (/\b(as a whole|in general|overall|at its core|in the end|when combined|taken together)\.?$/i.test(sentence)) score += 15;

  return Math.min(100, score);
}

// Correction 1: Sentence flattening — reduce one flourish in over-polished sentences.
const STYLO_FLOURISH_FLATTEN: [RegExp, string][] = [
  [/\btruly\s+/gi, ""],
  [/\bdeeply\s+/gi, ""],
  [/\bhighly\s+/gi, ""],
  [/\bquite\s+/gi, ""],
  [/\brather\s+/gi, ""],
  [/\bparticularly\s+/gi, ""],
  [/\bespecially\s+/gi, ""],
  [/\bfundamentally\s+/gi, ""],
  [/\bsignificantly\s+/gi, ""],
  [/\bremarkably\s+/gi, ""],
];

function styloFlattenSentence(sentence: string): string {
  // Apply only ONE flourish removal per sentence
  for (const [pattern, replacement] of STYLO_FLOURISH_FLATTEN) {
    if (pattern.test(sentence)) {
      const result = sentence.replace(pattern, replacement);
      // Fix double spaces and re-capitalize if needed
      const cleaned = result.replace(/\s{2,}/g, " ").trim();
      if (cleaned.length > 10) return cleaned;
    }
  }
  return sentence;
}

// Correction 2: Lexical simplification — replace one upgraded word.
const STYLO_SIMPLIFY: [RegExp, string][] = [
  [/\bdemonstrates?\b/gi, "shows"],
  [/\butiliz(e[sd]?|ing)\b/gi, "us$1"],
  [/\bfacilitat(e[sd]?|ing)\b/gi, "help$1"],
  [/\bimplement(ed|ing|s)?\b/gi, "set up"],
  [/\boptimiz(e[sd]?|ing)\b/gi, "improv$1"],
  [/\benhance[sd]?\b/gi, "improved"],
  [/\bexhibits?\b/gi, "shows"],
  [/\bpossess(es)?\b/gi, "has"],
  [/\bacquir(e[sd]?|ing)\b/gi, "get"],
  [/\bcommenc(e[sd]?|ing)\b/gi, "start"],
  [/\bconclude[sd]?\b/gi, "ended"],
  [/\bnumerous\b/gi, "many"],
  [/\bsubsequently\b/gi, "then"],
  [/\bprior to\b/gi, "before"],
  [/\bin the event that\b/gi, "if"],
];

function styloSimplify(sentence: string): string {
  for (const [pattern, replacement] of STYLO_SIMPLIFY) {
    if (pattern.test(sentence)) {
      return sentence.replace(pattern, replacement);
    }
  }
  return sentence;
}

// Correction 3: Punctuation disturbance — simplify balanced punctuation.
function styloPunctuationDisturb(sentence: string): string {
  // If sentence has a semicolon with balanced halves, replace with period + new sentence start
  const semiPos = sentence.indexOf(";");
  if (semiPos > 0) {
    const before = sentence.slice(0, semiPos).trim();
    const after = sentence.slice(semiPos + 1).trim();
    const beforeWords = before.split(/\s+/).length;
    const afterWords = after.split(/\s+/).length;
    // Balanced halves (within 40% of each other)
    const ratio = Math.min(beforeWords, afterWords) / Math.max(beforeWords, afterWords);
    if (ratio > 0.6 && afterWords > 3) {
      return before + ". " + after.charAt(0).toUpperCase() + after.slice(1);
    }
  }

  // If sentence has an em-dash pair creating a balanced aside, simplify to commas
  const emDashCount = (sentence.match(/—/g) || []).length;
  if (emDashCount === 2) {
    return sentence.replace(/—/g, ",");
  }

  return sentence;
}

// Correction 4: Cadence interruption — shorten one sentence if paragraph rhythm is too smooth.
function styloCadenceInterrupt(paragraph: string): string {
  const sentences = getSentences(paragraph);
  if (sentences.length < 4) return paragraph;

  const lengths = sentences.map(s => s.split(/\s+/).length);
  const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const variance = lengths.reduce((sum, l) => sum + Math.pow(l - avg, 2), 0) / lengths.length;
  const stdDev = Math.sqrt(variance);

  // Only interrupt if cadence is too smooth (low variance relative to avg)
  if (stdDev > avg * 0.35) return paragraph; // already varied enough

  // Find the longest mid-paragraph sentence and truncate it
  let maxLen = 0;
  let maxIdx = -1;
  for (let i = 1; i < sentences.length - 1; i++) {
    if (lengths[i] > maxLen) {
      maxLen = lengths[i];
      maxIdx = i;
    }
  }

  if (maxIdx < 0 || maxLen < 16) return paragraph;

  const sentence = sentences[maxIdx];
  // Truncate at first comma/semicolon after word 8
  const cutPoints = [...sentence.matchAll(/[,;]/g)];
  for (const cut of cutPoints) {
    if (cut.index && cut.index > 20 && cut.index < sentence.length - 15) {
      sentences[maxIdx] = sentence.slice(0, cut.index).trimEnd() + ".";
      return sentences.join(" ");
    }
  }

  return paragraph;
}

// Correction 5: Rhetorical bridge removal — if two rhetorical bridges appear within 3 sentences.
const RHETORICAL_BRIDGES = /\b(What this means|What stands out|The key takeaway|The bottom line|The point here|Put simply|Simply put|In short|In essence|At its core|The upshot)\b/i;

function styloRemoveExcessBridges(paragraph: string): string {
  const sentences = getSentences(paragraph);
  if (sentences.length < 3) return paragraph;

  let lastBridgeIdx = -10;
  for (let i = 0; i < sentences.length; i++) {
    if (RHETORICAL_BRIDGES.test(sentences[i])) {
      if (i - lastBridgeIdx <= 3) {
        // Two bridges within 3 sentences — strip the second one
        sentences[i] = sentences[i].replace(RHETORICAL_BRIDGES, "").trim();
        if (sentences[i].length > 5) {
          sentences[i] = sentences[i].charAt(0).toUpperCase() + sentences[i].slice(1);
          // Clean up leading punctuation artifacts
          sentences[i] = sentences[i].replace(/^[,;:]\s*/, "");
          if (sentences[i].length > 0) {
            sentences[i] = sentences[i].charAt(0).toUpperCase() + sentences[i].slice(1);
          }
        }
      }
      lastBridgeIdx = i;
    }
  }

  return sentences.join(" ");
}

// ─── Conversational Stylization Control ─────────────────────────────────────
// Limit strong rhetorical devices (fragment questions, emphatic pivots,
// ellipsis interruptions, rhetorical fragments) to max 1 per 4 sentences.
// Also limits emphasis channels: max 1 type (italics, dash, ellipsis, fragment)
// per paragraph to prevent stacked stylized texture.

const STRONG_DEVICE_RE = /(\?[^.!?]*$|^[A-Z][^.!?]{0,25}\?$|^(But|And|Yet|So) the (really |truly |most )?(interesting|tricky|hard|big|key|real) (part|bit|thing|question|issue)\b|^Here['']s the (thing|kicker|deal)|\.{3}|\u2026)/im;

function conversationalStylizationControl(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const result = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 4) return para;

    let lastDeviceIdx = -5;
    let modified = false;

    for (let i = 0; i < sentences.length; i++) {
      if (STRONG_DEVICE_RE.test(sentences[i])) {
        if (i - lastDeviceIdx < 4) {
          // Too close to previous device — flatten this one
          // Convert question to statement
          sentences[i] = sentences[i]
            .replace(/\?$/, ".")
            .replace(/^(But|And|Yet|So) the (really |truly |most )?(interesting|tricky|hard|big|key|real) (part|bit|thing|question|issue)/i, "One $4")
            .replace(/^Here['']s the (thing|kicker|deal)/i, "The point");
          // Strip ellipsis
          sentences[i] = sentences[i].replace(/\.{3}|\u2026/g, "—");
          modified = true;
        }
        lastDeviceIdx = i;
      }
    }

    if (!modified) return para;
    return sentences.join(" ");
  });

  return result.join("\n\n");
}

// ─── Emphasis Channel Limiter ────────────────────────────────────────────────
// If a paragraph already contains one emphasis channel (italics, dash
// interruption, ellipsis, or sentence fragment), flatten additional ones.
// Multiple emphasis channels create stylized texture detectors flag.

function emphasisChannelLimiter(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const result = paragraphs.map((para) => {
    let channels = 0;
    let result = para;

    const hasItalics = /\*[^*]+\*/.test(result);
    const hasDash = /\s—\s/.test(result);
    const hasEllipsis = /\.{3}|\u2026/.test(result);
    const hasFragment = /(?<=[.!?]\s)[A-Z][^.!?]{0,20}\.$/.test(result);

    channels = [hasItalics, hasDash, hasEllipsis, hasFragment].filter(Boolean).length;

    if (channels <= 1) return para;

    // Strip the second+ channel type found (preserve the first)
    let stripped = 0;
    if (hasItalics && stripped === 0) { stripped++; }
    else if (hasItalics) { result = result.replace(/\*([^*]+)\*/g, "$1"); }

    if (hasDash && stripped === 0) { stripped++; }
    else if (hasDash) { result = result.replace(/\s—\s/g, ", "); }

    if (hasEllipsis && stripped === 0) { stripped++; }
    else if (hasEllipsis) { result = result.replace(/\.{3}|\u2026/g, "."); }

    return result;
  });

  return result.join("\n\n");
}

// ─── Contrast Pattern Breaker ────────────────────────────────────────────────
// Detects repeated "statement → qualification → reflective correction" pattern
// across adjacent sentences and flattens one occurrence. Pattern:
//   sentence 1: positive/neutral claim
//   sentence 2: "but" / "however" / "yet" qualification
//   sentence 3: broader reflection ("this means" / "what matters" / "the real")
// If this pattern appears twice in one paragraph, strip the second qualification.

const QUALIFICATION_RE = /^(But|However|Yet|Still|That said|Then again|On the other hand),?\s/i;
const REFLECTION_RE = /^(This (means|suggests|shows|implies)|What (matters|counts|stands out)|The (real|true|key|broader|deeper))\b/i;

function contrastPatternBreaker(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const result = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 5) return para;

    let patternCount = 0;

    for (let i = 1; i < sentences.length - 1; i++) {
      const isQualification = QUALIFICATION_RE.test(sentences[i]);
      const isReflection = REFLECTION_RE.test(sentences[i + 1]);

      if (isQualification && isReflection) {
        patternCount++;
        if (patternCount >= 2) {
          // Flatten the second qualification: strip the connector
          const stripped = sentences[i].replace(QUALIFICATION_RE, "");
          if (stripped.length > 10) {
            sentences[i] = stripped.charAt(0).toUpperCase() + stripped.slice(1);
          }
          break;
        }
      }
    }

    return sentences.join(" ");
  });

  return result.join("\n\n");
}

// Master stylometric correction — applies corrections selectively based on polish score.
function stylometricCorrectionLayer(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const corrected = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;

    // Score each sentence
    const scores = sentences.map(s => sentencePolishScore(s));
    const avgPolish = scores.reduce((a, b) => a + b, 0) / scores.length;

    // Only apply corrections if average polish is high (>40)
    if (avgPolish <= 40) return para;

    // Apply sentence-level corrections to the most polished sentences
    let correctionCount = 0;
    const maxCorrections = Math.ceil(sentences.length * 0.35); // correct up to ~35%

    const processed = sentences.map((sentence, i) => {
      if (correctionCount >= maxCorrections) return sentence;
      if (scores[i] < 45) return sentence; // this sentence is fine

      let result = sentence;

      // Apply corrections in priority order — only one per sentence
      if (scores[i] >= 70) {
        // Very polished: flatten + simplify
        result = styloFlattenSentence(result);
        if (result !== sentence) { correctionCount++; return result; }
        result = styloSimplify(result);
        if (result !== sentence) { correctionCount++; return result; }
      } else if (scores[i] >= 55) {
        // Moderately polished: simplify or disturb punctuation
        result = styloSimplify(result);
        if (result !== sentence) { correctionCount++; return result; }
        result = styloPunctuationDisturb(result);
        if (result !== sentence) { correctionCount++; return result; }
      } else {
        // Mildly polished: just punctuation disturbance
        result = styloPunctuationDisturb(result);
        if (result !== sentence) { correctionCount++; return result; }
      }

      return result;
    });

    // Apply paragraph-level corrections
    let paragraphResult = processed.join(" ");
    paragraphResult = styloCadenceInterrupt(paragraphResult);
    paragraphResult = styloRemoveExcessBridges(paragraphResult);

    return paragraphResult;
  });

  return corrected.join("\n\n");
}

// ─── Sentence Signature Breaker ─────────────────────────────────────────────
// Phase 21: Detects consecutive sentences that share a similar "signature" —
// similar word count, similar punctuation shape, and similar rhetorical
// energy. When two adjacent sentences match on all three axes, one is
// mutated deterministically to break the local rhythm fingerprint.
//
// Signature = { length bucket, punctuation pattern, energy level }
// Two sentences "match" if all three components are identical.

type LengthBucket = "short" | "medium" | "long" | "very_long";
type PunctuationShape = "simple" | "one_break" | "multi_break" | "complex";
type EnergyLevel = "flat" | "moderate" | "elevated";

function lengthBucket(wordCount: number): LengthBucket {
  if (wordCount <= 10) return "short";
  if (wordCount <= 20) return "medium";
  if (wordCount <= 30) return "long";
  return "very_long";
}

function punctuationShape(sentence: string): PunctuationShape {
  const commas = (sentence.match(/,/g) || []).length;
  const semis = (sentence.match(/;/g) || []).length;
  const dashes = (sentence.match(/—/g) || []).length;
  const colons = (sentence.match(/:/g) || []).length;
  const total = commas + semis + dashes + colons;

  if (total === 0) return "simple";
  if (total === 1) return "one_break";
  if (total <= 3) return "multi_break";
  return "complex";
}

function energyLevel(sentence: string): EnergyLevel {
  let energy = 0;

  // Flourish/upgrade words add energy
  if (FLOURISH_MARKERS.test(sentence)) energy += 2;

  // Exclamation or question marks add energy
  if (/[!?]/.test(sentence)) energy += 1;

  // Intensifiers add energy
  if (/\b(very|really|truly|absolutely|completely|entirely|extremely)\b/i.test(sentence)) energy += 1;

  // Formal connectors add energy (elevated register)
  if (/\b(however|moreover|furthermore|nevertheless|consequently)\b/i.test(sentence)) energy += 1;

  if (energy === 0) return "flat";
  if (energy <= 2) return "moderate";
  return "elevated";
}

interface SentenceSignature {
  length: LengthBucket;
  punctuation: PunctuationShape;
  energy: EnergyLevel;
}

function getSignature(sentence: string): SentenceSignature {
  const words = sentence.split(/\s+/).length;
  return {
    length: lengthBucket(words),
    punctuation: punctuationShape(sentence),
    energy: energyLevel(sentence),
  };
}

function signaturesMatch(a: SentenceSignature, b: SentenceSignature): boolean {
  return a.length === b.length && a.punctuation === b.punctuation && a.energy === b.energy;
}

// Mutation strategies for breaking a matching signature:
// 1. If sentence has a comma, split at comma and keep first half (shorten)
// 2. If sentence starts with a connector, strip it
// 3. If sentence has a semicolon, convert to period (changes punctuation shape)
// 4. Prepend a short bridge word to shift length bucket
function mutateSentenceSignature(sentence: string): string {
  const words = sentence.split(/\s+/);

  // Strategy 1: If long enough and has comma, truncate at first comma after word 6
  if (words.length > 14) {
    const commaPos = sentence.indexOf(",", 15);
    if (commaPos > 0 && commaPos < sentence.length - 20) {
      return sentence.slice(0, commaPos).trimEnd() + ".";
    }
  }

  // Strategy 2: Strip leading connector to change energy
  const connectorMatch = sentence.match(/^(However|Moreover|Furthermore|Additionally|Nevertheless|Consequently|Indeed|Still|Yet),?\s+/i);
  if (connectorMatch) {
    const rest = sentence.slice(connectorMatch[0].length);
    return rest.charAt(0).toUpperCase() + rest.slice(1);
  }

  // Strategy 3: Convert semicolon to period
  const semiPos = sentence.indexOf(";");
  if (semiPos > 0) {
    const before = sentence.slice(0, semiPos).trimEnd();
    const after = sentence.slice(semiPos + 1).trimStart();
    if (after.length > 10) {
      return before + ". " + after.charAt(0).toUpperCase() + after.slice(1);
    }
  }

  // Strategy 4: If medium-length, make shorter by trimming trailing clause
  if (words.length >= 15 && words.length <= 25) {
    for (const pattern of TRAILING_EXTENSIONS) {
      if (pattern.test(sentence)) {
        const trimmed = sentence.replace(pattern, ".");
        if (trimmed.split(/\s+/).length >= 7) return trimmed;
      }
    }
  }

  return sentence; // no mutation available
}

function sentenceSignatureBreaker(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  let totalMutations = 0;
  const maxMutations = 4; // cap total mutations across entire text

  const result = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;

    const signatures = sentences.map(getSignature);
    let mutated = false;

    for (let i = 1; i < sentences.length; i++) {
      if (totalMutations >= maxMutations) break;
      if (mutated) break; // max one mutation per paragraph

      if (signaturesMatch(signatures[i - 1], signatures[i])) {
        // Mutate the second sentence of the matching pair
        const original = sentences[i];
        const changed = mutateSentenceSignature(original);
        if (changed !== original) {
          sentences[i] = changed;
          totalMutations++;
          mutated = true;
        }
      }
    }

    return sentences.join(" ");
  });

  return result.join("\n\n");
}

// ─── Micro Human Noise Engine ────────────────────────────────────────────────
// Phase 22: Introduces controlled, minimal irregularities that mimic human
// writing imperfection without harming readability.
//
// Three noise types — each applied at most once or twice per text:
// A) One shorter factual sentence injected into a paragraph that lacks one
// B) One restrained word repetition (echo a key noun from prior sentence)
// C) One flat/abrupt paragraph ending (strip tidy wrap-up)
//
// Never random chaos — each noise type is gated by structural analysis.

// Noise A: Ensure at least one paragraph has a noticeably short (<10 word)
// sentence. If no paragraph has one, pick the paragraph with the most
// uniform sentence lengths and insert a short bridging sentence.
const SHORT_BRIDGES = [
  "That part matters.",
  "This changed things.",
  "It shows.",
  "The difference is real.",
  "That stood out.",
  "Not everyone agrees.",
  "The data backs this up.",
  "Worth noting.",
  "Simple as that.",
  "And it works.",
];

function noiseShortSentence(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  // Check if any paragraph already has a very short sentence
  const hasShort = paragraphs.some((para) => {
    const sentences = getSentences(para);
    return sentences.some((s) => s.split(/\s+/).length <= 8);
  });

  if (hasShort) return text; // already has natural short sentence — skip

  // Find the most uniform paragraph (lowest length std dev)
  let bestIdx = -1;
  let lowestDev = Infinity;

  for (let i = 0; i < paragraphs.length; i++) {
    const sentences = getSentences(paragraphs[i]);
    if (sentences.length < 3) continue;

    const lengths = sentences.map((s) => s.split(/\s+/).length);
    const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    const dev = Math.sqrt(
      lengths.reduce((sum, l) => sum + Math.pow(l - avg, 2), 0) / lengths.length
    );

    if (dev < lowestDev) {
      lowestDev = dev;
      bestIdx = i;
    }
  }

  if (bestIdx < 0) return text;

  // Insert a short bridge sentence at position 2 (after second sentence)
  const sentences = getSentences(paragraphs[bestIdx]);
  if (sentences.length < 3) return text;

  const bridge = SHORT_BRIDGES[bestIdx % SHORT_BRIDGES.length];
  sentences.splice(2, 0, bridge);
  paragraphs[bestIdx] = sentences.join(" ");

  return paragraphs.join("\n\n");
}

// Noise B: Restrained word repetition.
// Humans naturally echo a key word from one sentence in the next.
// Find one place where two adjacent sentences share zero content words
// and echo one content word from the first into the second.
function noiseWordEcho(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  let applied = false;

  const result = paragraphs.map((para) => {
    if (applied) return para;

    const sentences = getSentences(para);
    if (sentences.length < 3) return para;

    for (let i = 0; i < sentences.length - 1; i++) {
      const wordsA = sentences[i]
        .toLowerCase()
        .replace(/[^a-z\s]/g, "")
        .split(/\s+/)
        .filter((w) => w.length > 4 && !STOP_WORDS.has(w));

      const wordsB = new Set(
        sentences[i + 1]
          .toLowerCase()
          .replace(/[^a-z\s]/g, "")
          .split(/\s+/)
      );

      // Check if zero content words overlap
      const overlap = wordsA.filter((w) => wordsB.has(w));
      if (overlap.length > 0) continue; // already has natural echo

      // Pick a content word from sentence A to echo
      if (wordsA.length === 0) continue;
      const echoWord = wordsA[Math.floor(wordsA.length / 2)];

      // Prepend an echo phrase to sentence B
      const nextSentence = sentences[i + 1];
      sentences[i + 1] = `That ${echoWord} ` +
        nextSentence.charAt(0).toLowerCase() + nextSentence.slice(1);

      // Fix: if the sentence already started with a capital word like "The", it's fine
      // but if it started with a proper noun, revert
      if (/^That \w+ [A-Z][a-z]/.test(sentences[i + 1])) {
        sentences[i + 1] = nextSentence; // revert — would sound awkward
        continue;
      }

      applied = true;
      break;
    }

    return sentences.join(" ");
  });

  return result.join("\n\n");
}

// Noise C: One flat paragraph ending.
// If the last sentence of a mid-text paragraph is long (>20 words) and
// contains a wrap-up pattern, replace it with a truncated version that
// ends abruptly. Applied to at most one paragraph.
const WRAPUP_PATTERNS = [
  /,?\s+which (ultimately|collectively|together|broadly|overall) .+\.$/i,
  /,?\s+demonstrating .+\.$/i,
  /,?\s+highlighting .+\.$/i,
  /,?\s+underscoring .+\.$/i,
  /,?\s+reflecting .+\.$/i,
  /,?\s+illustrating .+\.$/i,
  /,?\s+suggesting .+\.$/i,
  /,?\s+ensuring .+\.$/i,
];

function noiseFlatEnding(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  let applied = false;

  const result = paragraphs.map((para, idx) => {
    // Skip first and last paragraph
    if (applied || idx === 0 || idx === paragraphs.length - 1) return para;

    const sentences = getSentences(para);
    if (sentences.length < 2) return para;

    const lastSentence = sentences[sentences.length - 1];
    if (lastSentence.split(/\s+/).length < 18) return para;

    for (const pattern of WRAPUP_PATTERNS) {
      if (pattern.test(lastSentence)) {
        const flattened = lastSentence.replace(pattern, ".");
        if (flattened.split(/\s+/).length >= 6) {
          sentences[sentences.length - 1] = flattened;
          applied = true;
          return sentences.join(" ");
        }
      }
    }

    return para;
  });

  return result.join("\n\n");
}

// Master micro noise engine — applies noise types gated by register.
// Academic/formal: only flat endings (register-safe). Skip casual bridges and word echo.
function microHumanNoiseEngine(text: string, register: SourceRegister): string {
  let result = text;
  const isFormal = register === "academic" || register === "formal";

  if (!isFormal) {
    result = noiseShortSentence(result);
    result = noiseWordEcho(result);
  }
  result = noiseFlatEnding(result);
  return result;
}

// ─── First Paragraph Stylometric Hardening ──────────────────────────────────
// Phase 23: Detectors (especially GPTZero and Originality.ai) assign
// disproportionate confidence weight to the first paragraph. This pass
// runs extra stylometric corrections on paragraph 1 only:
// - Calmer first sentence (strip intensifiers, simplify vocabulary)
// - Lower lexical ambition (downgrade remaining upgrade words)
// - Reduced opener stylization (remove leading filler/connectors)

function firstParagraphStylometricHardening(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  if (paragraphs.length === 0) return text;

  let firstPara = paragraphs[0];
  const sentences = getSentences(firstPara);
  if (sentences.length === 0) return text;

  // A: Calm the first sentence
  let firstSentence = sentences[0];

  // Strip intensifiers from first sentence
  firstSentence = firstSentence
    .replace(/\b(truly|deeply|highly|remarkably|incredibly|extremely|particularly|especially|absolutely)\s+/gi, "")
    .replace(/\s{2,}/g, " ")
    .trim();

  // Simplify vocabulary in first sentence
  const firstSentenceSimplifications: [RegExp, string][] = [
    [/\btransformative\b/gi, "big"],
    [/\bsignificant\b/gi, "real"],
    [/\bsubstantial\b/gi, "solid"],
    [/\bcomprehensive\b/gi, "full"],
    [/\bfundamental\b/gi, "basic"],
    [/\bdemonstrates?\b/gi, "shows"],
    [/\bfacilitat(e[sd]?|ing)\b/gi, "help"],
    [/\bexhibits?\b/gi, "shows"],
    [/\bnumerous\b/gi, "many"],
    [/\bsophisticated\b/gi, "advanced"],
    [/\bphenomen(on|a)\b/gi, "thing"],
  ];
  for (const [pattern, replacement] of firstSentenceSimplifications) {
    firstSentence = firstSentence.replace(pattern, replacement);
  }

  // Ensure first sentence doesn't start with a stylized opener
  firstSentence = firstSentence
    .replace(/^(In today's world|In the modern era|Throughout history|In recent years|Across the globe|In an era of),?\s+/i, "")
    .trim();

  if (firstSentence.length > 5) {
    firstSentence = firstSentence.charAt(0).toUpperCase() + firstSentence.slice(1);
  }

  sentences[0] = firstSentence;

  // B: Lower lexical ambition across all sentences in first paragraph
  const lowerAmbition: [RegExp, string][] = [
    [/\bpivotal\b/gi, "key"],
    [/\bcrucial\b/gi, "important"],
    [/\bessential\b/gi, "needed"],
    [/\bvital\b/gi, "important"],
    [/\bprofound\b/gi, "real"],
    [/\bremarkable\b/gi, "clear"],
    [/\bextraordinary\b/gi, "unusual"],
    [/\bunprecedented\b/gi, "new"],
    [/\binnovative\b/gi, "new"],
    [/\bcompelling\b/gi, "strong"],
    [/\bnotable\b/gi, "clear"],
  ];

  for (let i = 1; i < sentences.length; i++) {
    let s = sentences[i];
    for (const [pattern, replacement] of lowerAmbition) {
      s = s.replace(pattern, replacement);
    }
    sentences[i] = s;
  }

  // C: If first paragraph has 4+ sentences, strip connector from sentence 2
  if (sentences.length >= 4) {
    const connectorRe = /^(However|Moreover|Furthermore|Additionally|Nevertheless|Indeed|Notably|Importantly|Critically|Crucially|Significantly),?\s+/i;
    if (connectorRe.test(sentences[1])) {
      const stripped = sentences[1].replace(connectorRe, "");
      if (stripped.length > 10) {
        sentences[1] = stripped.charAt(0).toUpperCase() + stripped.slice(1);
      }
    }
  }

  // D: If last sentence of first paragraph is a tidy summary, flatten it
  if (sentences.length >= 3) {
    const lastIdx = sentences.length - 1;
    let lastSentence = sentences[lastIdx];
    // Strip trailing wrap-up clauses
    lastSentence = lastSentence
      .replace(/,?\s+which (ultimately|essentially|fundamentally|clearly|broadly) .+\.$/i, ".")
      .replace(/,?\s+making (it|this|them) .+\.$/i, ".")
      .replace(/,?\s+and (this|that|these) (is|are|remain[s]?) .+\.$/i, ".");
    sentences[lastIdx] = lastSentence;
  }

  paragraphs[0] = sentences.join(" ");
  return paragraphs.join("\n\n");
}

// ─── Rhetorical Density Limiter ─────────────────────────────────────────────
// Phase 26: Counts conversational/rhetorical markers per paragraph and strips
// excess when density exceeds a threshold. LLM passes tend to over-inject
// casual markers to sound "human," creating a detectable over-distribution.
//
// Markers: "think about it", "look", "honestly", "ah", "you know", "for me",
// "I mean", "right?", "sure", "seriously", "let's be real", "the thing is"

const RHETORICAL_DENSITY_MARKERS: RegExp[] = [
  /\bthink about it\b/i,
  /\blook,?\s/i,
  /\bhonestly,?\s/i,
  /\bah,?\s/i,
  /\byou know,?\s/i,
  /\bfor me,?\s/i,
  /\bI mean,?\s/i,
  /\bright\?/i,
  /\bsure,?\s/i,
  /\bseriously,?\s/i,
  /\blet['']s be (real|honest)\b/i,
  /\bthe thing is,?\s/i,
  /\bto be fair,?\s/i,
  /\bif I['']m being honest\b/i,
  /\bisn['']t it\??/i,
  /\bdon['']t you think\??/i,
  /\bhere['']s the (thing|deal|kicker)\b/i,
  /\byou see,?\s/i,
  /\bbelieve it or not\b/i,
  /\btrust me\b/i,
];

function rhetoricalDensityLimiter(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);

  const result = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;

    // Count total markers in this paragraph
    let totalMarkers = 0;
    const markerLocations: { sentIdx: number; markerIdx: number }[] = [];

    for (let si = 0; si < sentences.length; si++) {
      for (let mi = 0; mi < RHETORICAL_DENSITY_MARKERS.length; mi++) {
        if (RHETORICAL_DENSITY_MARKERS[mi].test(sentences[si])) {
          totalMarkers++;
          markerLocations.push({ sentIdx: si, markerIdx: mi });
        }
      }
    }

    // Threshold: max 1 marker per 1.25 sentences (density ≤ 0.8).
    // Conversational markers are a human signal to external detectors —
    // only strip when it's extreme over-distribution (every sentence).
    const maxMarkers = Math.max(1, Math.floor(sentences.length * 0.8));
    if (totalMarkers <= maxMarkers) return para;

    // Strip excess markers from mid-paragraph sentences (keep first and last)
    let removed = 0;
    const targetRemove = totalMarkers - maxMarkers;

    for (const loc of markerLocations) {
      if (removed >= targetRemove) break;
      // Don't strip from first or last sentence
      if (loc.sentIdx === 0 || loc.sentIdx === sentences.length - 1) continue;

      const marker = RHETORICAL_DENSITY_MARKERS[loc.markerIdx];
      const original = sentences[loc.sentIdx];
      const stripped = original.replace(marker, "").replace(/^\s+/, "").replace(/\s{2,}/g, " ");

      if (stripped.length > 10) {
        sentences[loc.sentIdx] = stripped.charAt(0).toUpperCase() + stripped.slice(1);
        removed++;
      }
    }

    return sentences.join(" ");
  });

  return result.join("\n\n");
}

// ─── Lexical Spike Suppression ──────────────────────────────────────────────
// Phase 27: Detects clusters of upgraded/elevated vocabulary words appearing
// within a short span (100-word window). When 3+ elevated words cluster
// together, flattens the excess to prevent lexical confidence spikes that
// detectors interpret as model-generated "upgrade" behavior.

const ELEVATED_VOCAB: { re: RegExp; plain: string }[] = [
  { re: /\bcolossal\b/i,      plain: "huge" },
  { re: /\bmonumental\b/i,    plain: "big" },
  { re: /\bprofound\b/i,      plain: "deep" },
  { re: /\babsolutely\b/i,    plain: "fully" },
  { re: /\bincredibly\b/i,    plain: "very" },
  { re: /\bstartling\b/i,     plain: "surprising" },
  { re: /\bextraordinary\b/i, plain: "unusual" },
  { re: /\bimmensely\b/i,     plain: "very" },
  { re: /\bremarkably\b/i,    plain: "quite" },
  { re: /\bstunning\b/i,      plain: "striking" },
  { re: /\bspectacular\b/i,   plain: "impressive" },
  { re: /\bphenomenal\b/i,    plain: "strong" },
  { re: /\bmagnificent\b/i,   plain: "great" },
  { re: /\bexquisite\b/i,     plain: "fine" },
  { re: /\bformidable\b/i,    plain: "tough" },
  { re: /\bastounding\b/i,    plain: "surprising" },
  { re: /\bbreathtaking\b/i,  plain: "striking" },
  { re: /\bawe-inspiring\b/i, plain: "impressive" },
  { re: /\bimpeccable\b/i,    plain: "clean" },
  { re: /\bexceptional\b/i,   plain: "strong" },
  { re: /\bunparalleled\b/i,  plain: "rare" },
  { re: /\binvaluable\b/i,    plain: "useful" },
  { re: /\bindispensable\b/i, plain: "needed" },
  { re: /\bparamount\b/i,     plain: "top" },
  { re: /\bpivotal\b/i,       plain: "key" },
];

function lexicalSpikeSuppression(text: string): string {
  // Find all elevated word positions
  const hits: { index: number; length: number; vocabIdx: number }[] = [];

  for (let vi = 0; vi < ELEVATED_VOCAB.length; vi++) {
    const globalRe = new RegExp(ELEVATED_VOCAB[vi].re.source, "gi");
    let m: RegExpExecArray | null;
    while ((m = globalRe.exec(text)) !== null) {
      hits.push({ index: m.index, length: m[0].length, vocabIdx: vi });
    }
  }

  if (hits.length < 3) return text; // no spike possible
  hits.sort((a, b) => a.index - b.index);

  // Sliding window: find clusters of 3+ within ~100 word span (~600 chars)
  const WINDOW = 600;
  const toFlatten = new Set<number>(); // indices into hits[]

  for (let i = 0; i < hits.length; i++) {
    // Count how many hits fall within window starting at hits[i]
    const windowEnd = hits[i].index + WINDOW;
    const cluster: number[] = [];
    for (let j = i; j < hits.length && hits[j].index < windowEnd; j++) {
      cluster.push(j);
    }

    if (cluster.length >= 3) {
      // Flatten all but the first in the cluster
      for (let k = 1; k < cluster.length; k++) {
        toFlatten.add(cluster[k]);
      }
    }
  }

  if (toFlatten.size === 0) return text;

  // Apply replacements from end to start
  const sorted = [...toFlatten].sort((a, b) => hits[b].index - hits[a].index);
  let result = text;

  for (const hi of sorted) {
    const { index, length, vocabIdx } = hits[hi];
    const original = result.slice(index, index + length);
    let plain = ELEVATED_VOCAB[vocabIdx].plain;
    // Preserve capitalisation
    if (original[0] === original[0].toUpperCase()) {
      plain = plain.charAt(0).toUpperCase() + plain.slice(1);
    }
    result = result.slice(0, index) + plain + result.slice(index + length);
  }

  return result;
}

// ─── Semantic Variance Injector ─────────────────────────────────────────────
// Phase 28: Measures semantic similarity between consecutive paragraphs using
// a lightweight proxy: shared content-word ratio. If two adjacent paragraphs
// share too many content words (Jaccard > 0.25), inject local variance by:
//   A) Compress one explanation (shorten longest sentence)
//   B) Flatten one transition (strip connector from sentence 1 of second para)
//   C) Delay one detail (move last sentence to second-to-last position)
// Only one action per paragraph pair. Max 2 interventions per text.

function paragraphContentWords(para: string): Set<string> {
  return new Set(
    para
      .toLowerCase()
      .replace(/[^a-z\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 4 && !STOP_WORDS.has(w))
  );
}

function paragraphJaccard(a: Set<string>, b: Set<string>): number {
  let overlap = 0;
  for (const w of a) {
    if (b.has(w)) overlap++;
  }
  const union = new Set([...a, ...b]).size;
  return union === 0 ? 0 : overlap / union;
}

function semanticVarianceInjector(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  if (paragraphs.length < 3) return text;

  const contentWords = paragraphs.map(paragraphContentWords);
  let interventions = 0;

  for (let i = 1; i < paragraphs.length && interventions < 2; i++) {
    const similarity = paragraphJaccard(contentWords[i - 1], contentWords[i]);
    if (similarity < 0.25) continue; // sufficiently different

    const sentences = getSentences(paragraphs[i]);
    if (sentences.length < 3) continue;

    // Strategy A: Compress the longest sentence
    const lengths = sentences.map((s) => s.split(/\s+/).length);
    const maxIdx = lengths.indexOf(Math.max(...lengths));

    if (lengths[maxIdx] > 16) {
      const sentence = sentences[maxIdx];
      const commaPos = sentence.indexOf(",", 15);
      if (commaPos > 0 && commaPos < sentence.length - 15) {
        sentences[maxIdx] = sentence.slice(0, commaPos).trimEnd() + ".";
        paragraphs[i] = sentences.join(" ");
        interventions++;
        continue;
      }
    }

    // Strategy B: Strip connector from first sentence of second paragraph
    const connRe = /^(However|Moreover|Furthermore|Additionally|Nevertheless|Indeed|Similarly|Likewise|In addition|On top of that|Beyond this),?\s+/i;
    if (connRe.test(sentences[0])) {
      const stripped = sentences[0].replace(connRe, "");
      if (stripped.length > 10) {
        sentences[0] = stripped.charAt(0).toUpperCase() + stripped.slice(1);
        paragraphs[i] = sentences.join(" ");
        interventions++;
        continue;
      }
    }

    // Strategy C: Reorder — move last sentence to second-to-last position
    if (sentences.length >= 4) {
      const last = sentences.pop()!;
      sentences.splice(sentences.length - 1, 0, last);
      paragraphs[i] = sentences.join(" ");
      interventions++;
    }
  }

  return paragraphs.join("\n\n");
}

// ─── Register Profiler ──────────────────────────────────────────────────────
// Phase 29: Measures sentence register (plain/moderate/elevated) across the
// output and enforces mixed register. If all sentences in a paragraph sit at
// the same register level, mutate one to break tonal uniformity.
//
// Register classification:
//   plain    — short (<14 words), no fancy vocab, no formal connectors
//   moderate — medium length, some structure, everyday vocabulary
//   elevated — long, formal connectors, elevated vocabulary, complex clauses

type RegisterLevel = "plain" | "moderate" | "elevated";

const ELEVATED_WORDS_RE = /\b(consequently|nevertheless|furthermore|notwithstanding|subsequently|comprehensive|sophisticated|fundamental|significant|remarkable|extraordinary|unprecedented|indispensable|paradigm|transformative)\b/i;
const FORMAL_STRUCTURE_RE = /;|—|:|,\s+which\b|,\s+where\b|,\s+although\b/;

function classifySentenceRegister(sentence: string): RegisterLevel {
  const words = sentence.split(/\s+/).length;

  // Short plain sentences
  if (words <= 10 && !ELEVATED_WORDS_RE.test(sentence)) return "plain";

  // Check for elevated markers
  const hasElevatedVocab = ELEVATED_WORDS_RE.test(sentence);
  const hasFormalStructure = FORMAL_STRUCTURE_RE.test(sentence);
  const hasFormalConnector = /^(However|Moreover|Furthermore|Nevertheless|Consequently|Additionally)\b/i.test(sentence);

  if ((hasElevatedVocab && hasFormalStructure) || (hasFormalConnector && words > 20)) return "elevated";
  if (hasElevatedVocab || hasFormalStructure || words > 25) return "moderate";
  if (words <= 14) return "plain";

  return "moderate";
}

// Flatten an elevated sentence down to moderate
function flattenToModerate(sentence: string): string {
  let result = sentence;

  // Strip one formal connector
  result = result.replace(/^(However|Moreover|Furthermore|Nevertheless|Consequently|Additionally),?\s+/i, "");
  if (result.length > 5) {
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }

  // Replace one elevated word
  const elevatedMatch = result.match(ELEVATED_WORDS_RE);
  if (elevatedMatch) {
    const plainMap: Record<string, string> = {
      consequently: "so", nevertheless: "still", furthermore: "also",
      notwithstanding: "despite", subsequently: "then", comprehensive: "full",
      sophisticated: "complex", fundamental: "basic", significant: "real",
      remarkable: "clear", extraordinary: "unusual", unprecedented: "new",
      indispensable: "needed", paradigm: "model", transformative: "big",
    };
    const match = elevatedMatch[0].toLowerCase();
    const replacement = plainMap[match];
    if (replacement) {
      const preserve = elevatedMatch[0][0] === elevatedMatch[0][0].toUpperCase();
      const rep = preserve ? replacement.charAt(0).toUpperCase() + replacement.slice(1) : replacement;
      result = result.replace(ELEVATED_WORDS_RE, rep);
    }
  }

  return result;
}

// Elevate a plain sentence to moderate (add minor structure)
function elevateToModerate(sentence: string): string {
  // Only if very short — add a brief qualifying clause
  const words = sentence.split(/\s+/);
  if (words.length < 6) return sentence; // too short to modify safely

  // Remove trailing period, add a lightweight clause
  const QUALIFIERS = [
    ", at least in most cases.",
    ", or close to it.",
    ", broadly speaking.",
    ", in practical terms.",
    ", from what we can tell.",
  ];

  if (sentence.endsWith(".")) {
    const base = sentence.slice(0, -1);
    return base + QUALIFIERS[words.length % QUALIFIERS.length];
  }

  return sentence;
}

function registerProfiler(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  let totalAdjustments = 0;

  const result = paragraphs.map((para) => {
    if (totalAdjustments >= 3) return para; // cap total adjustments

    const sentences = getSentences(para);
    if (sentences.length < 3) return para;

    const registers = sentences.map(classifySentenceRegister);

    // Check if all sentences share the same register
    const allSame = registers.every((r) => r === registers[0]);
    if (!allSame) return para; // already mixed — good

    // All sentences at same register — only flatten elevated paragraphs.
    // Never elevate plain ones — adding qualifiers sounds more AI, not less.
    if (registers[0] === "elevated") {
      const midIdx = Math.floor(sentences.length / 2);
      sentences[midIdx] = flattenToModerate(sentences[midIdx]);
      totalAdjustments++;
    }
    // plain or moderate: leave alone — already acceptable

    return sentences.join(" ");
  });

  return result.join("\n\n");
}

// ─── Detector Fingerprint Analyzer ──────────────────────────────────────────
// Phase 30: Internal approximation of detector-sensitive signals. Computes
// a 5-axis fingerprint and applies targeted micro-corrections for the most
// problematic axis. Used as internal guidance only — does not overfit to any
// single external detector.
//
// Axes:
//   1. Opener confidence — first sentence complexity relative to rest
//   2. Consecutive rhythm similarity — adjacent sentence length match rate
//   3. Lexical upgrade clustering — elevated words per 100 words
//   4. Paragraph energy uniformity — std dev of paragraph word counts
//   5. Rhetorical density — conversational markers per 100 words

interface DetectorFingerprint {
  openerConfidence: number;    // 0-100 (lower = better)
  rhythmSimilarity: number;   // 0-100
  lexicalClustering: number;  // 0-100
  energyUniformity: number;   // 0-100
  rhetoricalDensity: number;  // 0-100
}

function computeDetectorFingerprint(text: string): DetectorFingerprint {
  const sentences = getSentences(text);
  const paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim().length > 0);
  const words = text.split(/\s+/);
  const wordCount = words.length;

  // 1. Opener confidence: first sentence word count vs average
  let openerConfidence = 50;
  if (sentences.length > 2) {
    const firstLen = sentences[0].split(/\s+/).length;
    const avgLen = sentences.slice(1).reduce((sum, s) => sum + s.split(/\s+/).length, 0) / (sentences.length - 1);
    const ratio = firstLen / Math.max(avgLen, 1);
    if (ratio > 0.85 && ratio < 1.15) openerConfidence = 75; // suspiciously similar
    else if (ratio > 1.3 || ratio < 0.7) openerConfidence = 20; // good variance
    else openerConfidence = 50;
  }

  // 2. Consecutive rhythm similarity: how many adjacent pairs have similar length
  let rhythmSimilarity = 50;
  if (sentences.length > 3) {
    let similarPairs = 0;
    for (let i = 1; i < sentences.length; i++) {
      const prevLen = sentences[i - 1].split(/\s+/).length;
      const currLen = sentences[i].split(/\s+/).length;
      const ratio = Math.min(prevLen, currLen) / Math.max(prevLen, currLen);
      if (ratio > 0.75) similarPairs++;
    }
    rhythmSimilarity = Math.round((similarPairs / (sentences.length - 1)) * 100);
  }

  // 3. Lexical upgrade clustering: elevated words per 100 words
  let lexicalClustering = 50;
  const elevatedCount = (text.match(/\b(profound|remarkable|extraordinary|transformative|pivotal|unprecedented|comprehensive|significant|sophisticated|fundamental|compelling|innovative)\b/gi) || []).length;
  const density = (elevatedCount / Math.max(wordCount, 1)) * 100;
  if (density > 3) lexicalClustering = 85;
  else if (density > 1.5) lexicalClustering = 65;
  else if (density > 0.5) lexicalClustering = 40;
  else lexicalClustering = 15;

  // 4. Paragraph energy uniformity: std dev of paragraph word counts
  let energyUniformity = 50;
  if (paragraphs.length >= 3) {
    const paraCounts = paragraphs.map((p) => p.split(/\s+/).length);
    const avg = paraCounts.reduce((a, b) => a + b, 0) / paraCounts.length;
    const stdDev = Math.sqrt(paraCounts.reduce((sum, c) => sum + Math.pow(c - avg, 2), 0) / paraCounts.length);
    const cv = stdDev / Math.max(avg, 1); // coefficient of variation
    if (cv < 0.15) energyUniformity = 80; // too uniform
    else if (cv < 0.25) energyUniformity = 55;
    else energyUniformity = 20; // good variance
  }

  // 5. Rhetorical density: casual/conversational markers per 100 words
  let rhetoricalDensity = 50;
  let markerCount = 0;
  for (const marker of RHETORICAL_DENSITY_MARKERS) {
    const matches = text.match(new RegExp(marker.source, "gi"));
    if (matches) markerCount += matches.length;
  }
  const markerDensity = (markerCount / Math.max(wordCount, 1)) * 100;
  if (markerDensity > 4) rhetoricalDensity = 80; // over-distributed
  else if (markerDensity > 2) rhetoricalDensity = 55;
  else if (markerDensity > 0.5) rhetoricalDensity = 30;
  else rhetoricalDensity = 15;

  return { openerConfidence, rhythmSimilarity, lexicalClustering, energyUniformity, rhetoricalDensity };
}

// Apply targeted correction for the worst axis
function detectorFingerprintCorrection(text: string): string {
  const fp = computeDetectorFingerprint(text);

  // Find the worst axis (highest score)
  const axes = [
    { name: "opener", score: fp.openerConfidence },
    { name: "rhythm", score: fp.rhythmSimilarity },
    { name: "lexical", score: fp.lexicalClustering },
    { name: "energy", score: fp.energyUniformity },
    { name: "rhetoric", score: fp.rhetoricalDensity },
  ];
  axes.sort((a, b) => b.score - a.score);
  const worst = axes[0];

  // Only correct if worst axis is problematic (>60)
  if (worst.score <= 60) return text;

  const paragraphs = text.split(/\n\s*\n/);

  switch (worst.name) {
    case "opener": {
      // Make first sentence noticeably different in length
      const sentences = getSentences(paragraphs[0]);
      if (sentences.length >= 2 && sentences[0].split(/\s+/).length > 12) {
        // Shorten first sentence
        const commaPos = sentences[0].indexOf(",", 10);
        if (commaPos > 0 && commaPos < sentences[0].length - 15) {
          sentences[0] = sentences[0].slice(0, commaPos).trimEnd() + ".";
          paragraphs[0] = sentences.join(" ");
        }
      }
      break;
    }
    case "rhythm": {
      // Find first pair of similar-length adjacent sentences and shorten one
      for (let pi = 0; pi < paragraphs.length; pi++) {
        const sentences = getSentences(paragraphs[pi]);
        if (sentences.length < 3) continue;
        for (let i = 1; i < sentences.length - 1; i++) {
          const prevLen = sentences[i - 1].split(/\s+/).length;
          const currLen = sentences[i].split(/\s+/).length;
          const ratio = Math.min(prevLen, currLen) / Math.max(prevLen, currLen);
          if (ratio > 0.8 && currLen > 14) {
            const commaPos = sentences[i].indexOf(",", 12);
            if (commaPos > 0 && commaPos < sentences[i].length - 12) {
              sentences[i] = sentences[i].slice(0, commaPos).trimEnd() + ".";
              paragraphs[pi] = sentences.join(" ");
              return paragraphs.join("\n\n");
            }
          }
        }
      }
      break;
    }
    case "energy": {
      // Shorten the longest paragraph by truncating its longest sentence
      const paraCounts = paragraphs.map((p) => p.split(/\s+/).length);
      const maxIdx = paraCounts.indexOf(Math.max(...paraCounts));
      if (maxIdx >= 0) {
        const sentences = getSentences(paragraphs[maxIdx]);
        if (sentences.length >= 3) {
          const lengths = sentences.map((s) => s.split(/\s+/).length);
          const longestIdx = lengths.indexOf(Math.max(...lengths));
          if (lengths[longestIdx] > 16 && longestIdx > 0) {
            const s = sentences[longestIdx];
            const commaPos = s.indexOf(",", 15);
            if (commaPos > 0 && commaPos < s.length - 12) {
              sentences[longestIdx] = s.slice(0, commaPos).trimEnd() + ".";
              paragraphs[maxIdx] = sentences.join(" ");
            }
          }
        }
      }
      break;
    }
    // lexical and rhetoric axes are already handled by earlier passes (27, 26)
    default:
      break;
  }

  return paragraphs.join("\n\n");
}

// ─── Light Factual Grounding Layer ──────────────────────────────────────────
// Phase 31: Reduces generic abstraction density by replacing vague
// generalizations with lightweight grounding phrases. Does NOT fabricate
// facts or add fake references — only makes existing claims sound more
// grounded and less generalized.
//
// Pattern: detect vague universal claims and inject grounding qualifiers.

// Only safe single-word swaps that reduce over-confident universals.
// Removed all phrase rewrites that added formal hedges — those made
// text sound more AI (academic hedging = AI signal to QuillBot/ZeroGPT).
const VAGUE_TO_GROUNDED: [RegExp, string][] = [
  [/\balways leads to\b/gi, "often leads to"],
  [/\balways results in\b/gi, "typically results in"],
  [/\bnever works\b/gi, "rarely works"],
  [/\bnever succeeds\b/gi, "rarely succeeds"],
  [/\bsince time immemorial\b/gi, "for generations"],
  [/\bsince the beginning of time\b/gi, "for a long time"],
];

function lightFactualGrounding(text: string): string {
  let result = text;
  let replacements = 0;
  const maxReplacements = 4; // don't over-ground — keep it light

  for (const [pattern, replacement] of VAGUE_TO_GROUNDED) {
    if (replacements >= maxReplacements) break;

    const before = result;
    result = result.replace(pattern, (match, ...args) => {
      if (replacements >= maxReplacements) return match;
      replacements++;
      // Handle capture groups in replacement string
      let rep = replacement;
      for (let i = 0; i < args.length - 2; i++) {
        if (typeof args[i] === "string") {
          rep = rep.replace(`$${i + 1}`, args[i]);
        }
      }
      return rep;
    });

    if (result !== before) {
      // Count how many replacements this pattern made
      // (already tracked in the replacer above)
    }
  }

  return result;
}

// ─── Adaptive Aggression Ceiling ─────────────────────────────────────────────
// Phase 25: If internal detector score is already below threshold after the
// core deterministic passes, reduce or skip the strongest later passes to
// avoid over-humanizing already-safe text. Over-humanization introduces its
// own detectable patterns (too many contractions, too casual, too fragmented).
//
// Strategy: score the text at a mid-pipeline checkpoint. If score < 30,
// skip the heaviest surgery passes (micro-surgery, signature breaker, noise
// engine). If score < 20, also skip stylometric correction.

type AggressionLevel = "full" | "reduced" | "minimal";

function determineAggressionLevel(text: string): AggressionLevel {
  const { score } = detectAI(text);
  if (score < 20) return "minimal";
  if (score < 30) return "reduced";
  return "full";
}

// ─── Public API ─────────────────────────────────────────────────────────────

export async function humanize(
  inputText: string,
  mode: HumanizeMode,
  wordLimit: number
): Promise<string> {
  const truncated = truncateToWordLimit(inputText, wordLimit);
  const inputWordCount = truncated.split(/\s+/).length;

  // Classify source register BEFORE any transformation — acts as hard ceiling
  const register = classifyRegister(truncated);

  // Split into variable-size chunks for natural local inconsistency
  const chunks = splitIntoVariableChunks(truncated);

  // Pass 1: Structural rewrite — parallel per chunk
  const structural = await structuralPass(chunks, mode, register);

  // Pass 2: Semantic naturalness — parallel per chunk (skipped for light)
  const semantic = await semanticPass(structural, mode, register);

  // Merge chunks back into full text
  const merged = semantic.join("\n\n");

  // Pass 3: Targeted mutation on full merged text (gated by score)
  const mutated = await mutationPass(merged, mode);

  // Pass 3a: Short-text perplexity hardening (LLM call, only fires for <120 word texts)
  const shortHardened = await shortTextPerplexityHardening(mutated, mode, register);

  // Pass 3b: First-paragraph hardening (extra mutation if opener still scores high)
  const hardened = await firstParagraphHardening(shortHardened, mode);

  // Pass 4: Deterministic anti-pattern cleanup (no LLM call)
  const cleaned = antiPatternPass(hardened, register);

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

  // Pass 8a: Rhetorical density limiter (no LLM call) — cap conversational markers
  const rhetoricLimited = rhetoricalDensityLimiter(detectorHardened);

  // Pass 8a2: Lexical spike suppression (no LLM call) — flatten vocabulary clusters
  const spikeFlattened = lexicalSpikeSuppression(rhetoricLimited);

  // Pass 8a3: Semantic variance injector (no LLM call) — reduce inter-paragraph similarity
  const semanticVaried = semanticVarianceInjector(spikeFlattened);

  // Pass 8a4: Register profiler (no LLM call) — enforce mixed sentence register
  const registerMixed = registerProfiler(semanticVaried);

  // Pass 8a5: Detector fingerprint analyzer (no LLM call) — targeted worst-axis correction
  const fingerprintCorrected = detectorFingerprintCorrection(registerMixed);

  // Pass 8a6: Light factual grounding (no LLM call) — reduce generic abstractions
  const grounded = lightFactualGrounding(fingerprintCorrected);

  // ── Adaptive aggression checkpoint ──
  // Score the text after core passes to determine if later heavy passes are needed.
  const aggression = determineAggressionLevel(grounded);

  // Pass 8b: Deterministic micro-surgery (no LLM call) — skip if already safe
  const surgeryResult = aggression === "minimal"
    ? grounded
    : deterministicMicroSurgery(grounded);

  // Pass 8c: Multi-detector hardening (no LLM call)
  const multiHardened = aggression === "minimal"
    ? surgeryResult
    : multiDetectorHardening(surgeryResult, register);

  // Pass 8d: Perplexity injector (no LLM call) — always runs, ZeroGPT targeted
  // Swaps one common word per paragraph with a lower-frequency synonym to
  // raise per-token unpredictability without changing meaning or register.
  const perplexityHardened = perplexityInjector(multiHardened, register);

  // Pass 9: Low-mutation islands (no LLM call) — always runs (preserves natural feel)
  const islanded = lowMutationIslands(perplexityHardened, truncated);

  // Pass 10: Length discipline — always runs
  const lengthEnforced = enforceLengthDiscipline(islanded, inputWordCount);

  // Pass 11: Stylometric correction layer (no LLM call) — skip if minimal aggression
  const styloCorrected = aggression === "minimal"
    ? lengthEnforced
    : stylometricCorrectionLayer(lengthEnforced);

  // Pass 12: Sentence signature breaker (no LLM call) — skip if reduced or minimal
  const signatureBroken = aggression === "full"
    ? sentenceSignatureBreaker(styloCorrected)
    : styloCorrected;

  // Pass 13: Micro human noise engine (no LLM call) — skip if reduced or minimal
  const noised = aggression === "full"
    ? microHumanNoiseEngine(signatureBroken, register)
    : signatureBroken;

  // Pass 13a: Conversational stylization control (no LLM call) — cap rhetorical devices
  const stylizationControlled = conversationalStylizationControl(noised);

  // Pass 13b: Emphasis channel limiter (no LLM call) — max 1 emphasis type per paragraph
  const emphasisLimited = emphasisChannelLimiter(stylizationControlled);

  // Pass 13c: Contrast pattern breaker (no LLM call) — break statement→qualification→reflection
  const contrastBroken = contrastPatternBreaker(emphasisLimited);

  // Pass 14: First paragraph stylometric hardening (no LLM call) — always runs (lightweight)
  return firstParagraphStylometricHardening(contrastBroken);
}
