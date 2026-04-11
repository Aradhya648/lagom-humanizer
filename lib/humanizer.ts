import { GoogleGenerativeAI } from "@google/generative-ai";
import { detectAI } from "@/lib/detector";
import {
  getStructuralPrompt,
  getSemanticPrompt,
  getMutationPrompt,
  type ContentType,
} from "@/prompts/pipeline";

// ─── Source Register ──────────────────────────────────────────────────────────
// Kept as a local type and function — antiPatternPass still uses it.

export type SourceRegister = "academic" | "formal" | "neutral" | "informal";

const ACADEMIC_MARKERS = /\b(furthermore|moreover|consequently|notwithstanding|henceforth|thereby|wherein|thus|hence|herein|aforementioned|et al|i\.e\.|e\.g\.)\b/i;
const FORMAL_MARKERS = /\b(therefore|however|nevertheless|regarding|pertaining|respectively|accordingly|subsequent|preceding)\b/i;
const INFORMAL_MARKERS = /\b(gonna|wanna|gotta|kinda|sorta|yeah|nah|okay|ok|hey|yep|nope|stuff|thing is|you know|I mean|right\?|honestly)\b/i;
const CONTRACTION_RE = /\b(I'm|you're|we're|they're|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't|it's|that's|there's|here's|what's|who's|let's|I've|you've|we've|they've|I'll|you'll|we'll|they'll|I'd|you'd|we'd|they'd)\b/gi;

export function classifyRegister(text: string): SourceRegister {
  const words = text.split(/\s+/).length;
  const sentences = text.split(/(?<=[.!?])\s+/).length;
  const avgSentLen = words / Math.max(sentences, 1);

  const academicHits = (text.match(ACADEMIC_MARKERS) || []).length;
  const formalHits = (text.match(FORMAL_MARKERS) || []).length;
  const informalHits = (text.match(INFORMAL_MARKERS) || []).length;
  const contractionCount = (text.match(CONTRACTION_RE) || []).length;
  const contractionDensity = contractionCount / Math.max(words, 1) * 100;

  if (contractionDensity > 2.5 || informalHits >= 3) return "informal";
  if (informalHits >= 2 && contractionDensity > 1.5) return "informal";
  if (academicHits >= 2) return "academic";
  if (academicHits >= 1 && avgSentLen > 22) return "academic";
  if (formalHits >= 3 && avgSentLen > 20) return "academic";
  if (formalHits >= 2 && contractionDensity < 1) return "formal";
  if (avgSentLen > 20 && contractionDensity < 0.5) return "formal";
  return "neutral";
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function truncateToWordLimit(text: string, wordLimit: number): string {
  const words = text.trim().split(/\s+/);
  if (words.length <= wordLimit) return text.trim();
  return words.slice(0, wordLimit).join(" ");
}

function getSentences(para: string): string[] {
  return para
    .split(/(?<=[.!?])\s+(?=[A-Z"'])/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

export function splitIntoVariableChunks(text: string): string[] {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const raw: string[] = [];

  for (const para of paragraphs) {
    const sentences = getSentences(para);

    if (sentences.length <= 3) {
      raw.push(para);
      continue;
    }

    let i = 0;
    while (i < sentences.length) {
      const wordCount = sentences[i].split(/\s+/).length;
      const groupSize = wordCount > 22 ? 2 : i % 2 === 0 ? 3 : 2;
      const end = Math.min(i + groupSize, sentences.length);
      raw.push(sentences.slice(i, end).join(" "));
      i = end;
    }
  }

  const chunks: string[] = [];
  for (const chunk of raw) {
    if (chunks.length > 0 && chunk.split(/\s+/).length < 15) {
      chunks[chunks.length - 1] += " " + chunk;
    } else {
      chunks.push(chunk);
    }
  }

  if (chunks.length > 8) {
    const head = chunks.slice(0, 7);
    const tail = chunks.slice(7).join(" ");
    return [...head, tail];
  }

  return chunks.length > 0 ? chunks : [text.trim()];
}

// ─── Sentence Classification ──────────────────────────────────────────────────

const SENTENCE_FILLER_RE = /\b(it is important to note|it is worth noting|furthermore|moreover|additionally|consequently|in conclusion|it is crucial|it is essential|delve into|in today's world|plays a crucial role|plays a vital role)\b/i;
const SENTENCE_FORMAL_OPENER_RE = /^(However|Moreover|Furthermore|Additionally|Consequently|Nevertheless|It is important|It is worth|In conclusion|In summary)\b/i;

export type SentenceClass = "A" | "B" | "C";

export function classifySentence(
  sentence: string,
  avgNeighborLength?: number
): SentenceClass {
  const words = sentence.split(/\s+/).length;
  if (words <= 10) return "A";
  if (SENTENCE_FILLER_RE.test(sentence)) return "C";
  if (SENTENCE_FORMAL_OPENER_RE.test(sentence)) return "C";
  if (avgNeighborLength !== undefined) {
    const ratio = words / avgNeighborLength;
    if (ratio > 0.85 && ratio < 1.15) return "B";
  }
  if (words <= 16) return "A";
  return "B";
}

export function annotateChunkWithClasses(chunk: string): {
  annotated: string;
  classes: SentenceClass[];
} {
  const sentences = getSentences(chunk);
  if (sentences.length === 0) return { annotated: chunk, classes: [] };

  const lengths = sentences.map((s) => s.split(/\s+/).length);
  const classes: SentenceClass[] = sentences.map((s, i) => {
    const neighbors = lengths.filter((_, j) => j !== i && Math.abs(j - i) <= 1);
    const avgNeighbor =
      neighbors.length > 0
        ? neighbors.reduce((a, b) => a + b, 0) / neighbors.length
        : undefined;
    return classifySentence(s, avgNeighbor);
  });

  const annotated = sentences.map((s, i) => `[${classes[i]}] ${s}`).join("\n");
  return { annotated, classes };
}

// ─── Generation Settings ──────────────────────────────────────────────────────

interface GenSettings {
  temperature: number;
  topP: number;
  topK?: number;
}

const STRUCTURAL_SETTINGS: GenSettings = { temperature: 0.85, topP: 0.95 };
const SEMANTIC_SETTINGS: GenSettings  = { temperature: 0.75, topP: 0.90 };
const MUTATION_SETTINGS: GenSettings  = { temperature: 0.88, topP: 0.92 };

// ─── Model Wrappers ───────────────────────────────────────────────────────────

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

async function callModel(prompt: string, settings: GenSettings): Promise<string> {
  return callGemini(prompt, settings);
}

// ─── Pipeline Passes ──────────────────────────────────────────────────────────

function validateChunkOutput(output: string, fallback: string): string {
  return output.trim().length > 0 ? output.trim() : fallback;
}

// Pass 1 — Structural rewrite per chunk (sequential, 3s delay between calls).
async function structuralPass(
  chunks: string[],
  contentType: ContentType
): Promise<string[]> {
  const results: string[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const out = await callModel(
      getStructuralPrompt(chunks[i], contentType, i),
      STRUCTURAL_SETTINGS
    ).then(o => validateChunkOutput(o, chunks[i]))
     .catch(() => chunks[i]);
    results.push(out);
    if (i < chunks.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 3000));
    }
  }
  return results;
}

// Pass 2 — Semantic naturalness per chunk (sequential, 3s delay between calls).
async function semanticPass(
  chunks: string[],
  contentType: ContentType
): Promise<string[]> {
  const results: string[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const out = await callModel(
      getSemanticPrompt(chunks[i], contentType, i),
      SEMANTIC_SETTINGS
    ).then(o => validateChunkOutput(o, chunks[i]))
     .catch(() => chunks[i]);
    results.push(out);
    if (i < chunks.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 3000));
    }
  }
  return results;
}

// Pass 3 — Selective mutation on full merged text. Gated by score > 45.
async function mutationPass(
  text: string,
  contentType: ContentType
): Promise<string> {
  return callModel(getMutationPrompt(text, contentType), MUTATION_SETTINGS);
}

// ─── Anti-Pattern Destruction ─────────────────────────────────────────────────
// Catches AI-signpost paragraph openers that models re-introduce despite
// prompt-level suppression.

const ANTI_PATTERNS: { regex: RegExp; replacements: string[] }[] = [
  { regex: /^Of course,?\s*/im,         replacements: ["", "Sure — ", "Look, ", "Right — "] },
  { regex: /^Now,?\s*/im,               replacements: ["", "So ", "At this point, ", "Then again, "] },
  { regex: /^Here'?s the thing:?\s*/im, replacements: ["", "The thing is, ", "What matters here: ", "Put simply, "] },
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

const CASUAL_REPLACEMENT_RE = /^(Sure|Look|Right|So\b|Then again|All told|Plus|The thing is|One thing that stands out|What matters|Worth flagging)/i;

function antiPatternPass(text: string, register: SourceRegister): string {
  const paragraphs = text.split(/\n\s*\n/);
  let replacementIndex = 0;
  const isFormal = register === "academic" || register === "formal";

  const cleaned = paragraphs.map((para) => {
    let result = para;
    for (const { regex, replacements } of ANTI_PATTERNS) {
      if (regex.test(result)) {
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

// ─── Rhetorical Fluency Suppression ──────────────────────────────────────────
// Replaces genuine AI-overuse multi-word patterns with grounded alternatives.
// Single-word substitutions deliberately excluded — those damage output quality.

const RHETORICAL_REPLACEMENTS: [RegExp, string][] = [
  [/\btruly incredible\b/gi,             "notable"],
  [/\bdeeply layered\b/gi,               "layered"],
  [/\butterly rooted\b/gi,               "rooted"],
  [/\bexact same moment\b/gi,            "same moment"],
  [/\btruly transformative\b/gi,         "significant"],
  [/\bdeeply rooted\b/gi,               "rooted"],
  [/\bprofoundly important\b/gi,         "important"],
  [/\bimmensely powerful\b/gi,           "powerful"],
  [/\bfundamentally reshape\b/gi,        "reshape"],
  [/\bparadigm shift\b/gi,              "major shift"],
  [/\bgroundbreaking research\b/gi,      "new research"],
  [/\bcutting[- ]edge technology\b/gi,   "modern technology"],
  [/\bstate[- ]of[- ]the[- ]art\b/gi,   "current"],
  [/\bgame[- ]changing\b/gi,            "significant"],
  [/\binextricably linked\b/gi,          "closely linked"],
  [/\boverarchingly\b/gi,               "broadly"],
];

function rhetoricalSuppressionPass(text: string): string {
  let result = text;
  for (const [pattern, replacement] of RHETORICAL_REPLACEMENTS) {
    result = result.replace(pattern, replacement);
  }
  return result;
}

// ─── Length Discipline ────────────────────────────────────────────────────────

function enforceLengthDiscipline(output: string, inputWordCount: number): string {
  const maxWords = Math.ceil(inputWordCount * 1.10);
  const outputWords = output.split(/\s+/).length;

  if (outputWords <= maxWords) return output;

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

// ─── Public API ───────────────────────────────────────────────────────────────

export async function humanize(
  inputText: string,
  contentType: ContentType,
  wordLimit: number
): Promise<string> {
  const truncated = truncateToWordLimit(inputText, wordLimit);
  const inputWordCount = truncated.split(/\s+/).length;

  const register = classifyRegister(truncated);
  const chunks = splitIntoVariableChunks(truncated);

  // Pass 1: Structural rewrite — parallel per chunk
  const structural = await structuralPass(chunks, contentType);

  // Pass 2: Semantic naturalness — parallel per chunk (all content types)
  const semantic = await semanticPass(structural, contentType);

  // Merge chunks
  const merged = semantic.join("\n\n");

  // Pass 3: Targeted mutation on full text — only if score > 20
  const { score } = detectAI(merged);
  const mutated = score > 20
    ? await mutationPass(merged, contentType)
    : merged;

  // Pass 4: Anti-pattern cleanup
  const cleaned = antiPatternPass(mutated, register);

  // Pass 5: Rhetorical fluency suppression
  const suppressed = rhetoricalSuppressionPass(cleaned);

  // Pass 6: Length discipline
  return enforceLengthDiscipline(suppressed, inputWordCount);
}
