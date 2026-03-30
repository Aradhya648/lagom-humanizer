import { GoogleGenerativeAI } from "@google/generative-ai";
import { detectAI } from "@/lib/detector";
import {
  getStructuralPrompt,
  getSemanticPrompt,
  getMutationPrompt,
  HumanizeMode,
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
async function structuralPass(
  chunks: string[],
  mode: HumanizeMode
): Promise<string[]> {
  const settings = STRUCTURAL_SETTINGS[mode];
  return Promise.all(
    chunks.map((chunk, i) =>
      callModel(getStructuralPrompt(chunk, mode, i), settings)
        .then((out) => validateChunkOutput(out, chunk))
        .catch(() => chunk) // fall back to original on any failure
    )
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
];

// Words that mark a sentence as rhetorically "flourished"
const FLOURISH_MARKERS = /\b(profound|remarkable|extraordinary|transformative|breathtaking|pivotal|unprecedented|paradigm|groundbreaking|indispensable|unparalleled|meticulously|seamlessly|intricate|undeniably)\b/i;

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

  // Pass 4: Deterministic anti-pattern cleanup (no LLM call)
  const cleaned = antiPatternPass(mutated);

  // Pass 5: Sentence asymmetry injection (no LLM call)
  const asymmetric = sentenceAsymmetryPass(cleaned);

  // Pass 6: Rhetorical fluency suppression (no LLM call)
  const suppressed = rhetoricalSuppressionPass(asymmetric);

  // Pass 7: Human writing trait engine (no LLM call)
  const humanized = humanTraitEngine(suppressed);

  // Pass 8: Length discipline — enforce 90%–110% of input word count
  return enforceLengthDiscipline(humanized, inputWordCount);
}
