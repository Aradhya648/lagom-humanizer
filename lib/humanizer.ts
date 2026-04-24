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

// Passive voice is a strong formal signal even without trigger words
const PASSIVE_VOICE_RE = /\b(is|are|was|were|be|been|being)\s+\w+ed\b/gi;
// Third-person impersonal constructions — formal writing pattern
const THIRD_PERSON_RE = /\b(one|a person|people|individuals|researchers|scholars|the author|the study|the data)\b/gi;

export function classifyRegister(text: string): SourceRegister {
  const words = text.split(/\s+/).length;
  const sentences = text.split(/(?<=[.!?])\s+/).length;
  const avgSentLen = words / Math.max(sentences, 1);

  const academicHits = (text.match(ACADEMIC_MARKERS) || []).length;
  const formalHits = (text.match(FORMAL_MARKERS) || []).length;
  const informalHits = (text.match(INFORMAL_MARKERS) || []).length;
  const contractionCount = (text.match(CONTRACTION_RE) || []).length;
  const contractionDensity = contractionCount / Math.max(words, 1) * 100;
  const passiveCount = (text.match(PASSIVE_VOICE_RE) || []).length;
  const passiveDensity = passiveCount / Math.max(sentences, 1);
  const thirdPersonHits = (text.match(THIRD_PERSON_RE) || []).length;

  if (contractionDensity > 2.5 || informalHits >= 3) return "informal";
  if (informalHits >= 2 && contractionDensity > 1.5) return "informal";

  if (academicHits >= 2) return "academic";
  if (academicHits >= 1 && avgSentLen > 22) return "academic";
  if (formalHits >= 3 && avgSentLen > 20) return "academic";

  if (formalHits >= 2 && contractionDensity < 1) return "formal";
  if (avgSentLen > 20 && contractionDensity < 0.5) return "formal";

  // Catch formal text that uses no trigger words:
  // Zero contractions + passive voice + third-person impersonal = formal writing
  if (contractionDensity === 0 && passiveDensity >= 0.3 && thirdPersonHits >= 2) return "formal";
  // Zero contractions + long sentences + third-person = formal
  if (contractionDensity === 0 && avgSentLen >= 16 && thirdPersonHits >= 3) return "formal";
  // Zero contractions + long sentences, no informal markers = at least formal
  if (contractionDensity === 0 && informalHits === 0 && avgSentLen >= 18) return "formal";

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
  maxOutputTokens?: number;
}

// Lowered temperatures: high temps were making Gemini produce ornate,
// verbose "creative" output. Lower = more deterministic, plainer language.
const STRUCTURAL_SETTINGS: GenSettings = { temperature: 0.60, topP: 0.90, maxOutputTokens: 4096 };
const SEMANTIC_SETTINGS: GenSettings  = { temperature: 0.55, topP: 0.88, maxOutputTokens: 4096 };
const MUTATION_SETTINGS: GenSettings  = { temperature: 0.60, topP: 0.88, maxOutputTokens: 4096 };
const GEMINI_TIMEOUT_MS = 30_000;
const SHORT_FAST_PATH_WORDS = 120;
const SINGLE_PASS_SETTINGS: GenSettings = { temperature: 0.60, topP: 0.88, maxOutputTokens: 4096 };
const FAST_MODEL = "gemini-2.5-flash";

// Flash is materially faster here and the quality delta is negligible in fast mode.
const MUTATION_MODEL = FAST_MODEL;

// ─── Model Wrappers ───────────────────────────────────────────────────────────

async function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  const timeout = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);
  });

  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

async function callGemini(prompt: string, settings: GenSettings): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: FAST_MODEL,
    generationConfig: {
      temperature: settings.temperature,
      topP: settings.topP,
      topK: settings.topK,
      maxOutputTokens: settings.maxOutputTokens ?? 4096,
    },
  });

  const result = await withTimeout(
    model.generateContent(prompt),
    GEMINI_TIMEOUT_MS,
    "Gemini flash call"
  );
  const text = result.response.text();
  if (!text || text.trim().length === 0) {
    throw new Error("Gemini returned empty response");
  }
  return text.trim();
}

async function callModel(prompt: string, settings: GenSettings): Promise<string> {
  return callGemini(prompt, settings);
}

// ─── NVIDIA NIM — Tier 0 Fingerprint Break ────────────────────────────────────
// NVIDIA NIM exposes a free OpenAI-compatible inference API at
// https://integrate.api.nvidia.com/v1. Running Gemini output through a
// completely different model family shatters the Gemini statistical signature
// that detectors (especially GPTZero) are trained to recognise.
//
// Set NVIDIA_API_KEY in your environment.
// Optionally set NVIDIA_MODEL to override the default (e.g. "minimaxai/minimax-m2.7").
// If NVIDIA_API_KEY is absent the pass is silently skipped — zero regression risk.

const NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1";
// Live-benchmarked against this actual account — 500-word chunk finishes in ~2s.
// 15s timeout is generous; fail-fast if NIM is cold rather than stalling the pipeline.
const NVIDIA_TIMEOUT_MS = 15_000;
// ─── MODEL CHOICE: openai/gpt-oss-20b ────────────────────────────────────────
// Winner from live benchmark across all models accessible on this free account:
//
//   • 2.2s end-to-end for a 100-word chunk (fastest working model on this account)
//   • GPT lineage — trained to produce natural, human-sounding prose by default.
//     This is exactly what we need: fingerprint-breaking + naturalness in one pass.
//   • 20B parameters — enough capacity to follow complex rewriting constraints
//   • Accessible on this free-tier account (unlike minitron-8b, mistral-nemo-12b,
//     palmyra-creative — all returned 404 on this account tier)
//
// Live benchmark results (real latency on this account, not theoretical):
//   openai/gpt-oss-20b                     → ✅  2.2s  — WINNER
//   meta/llama-3.2-3b-instruct             → ✅  2.1s  — fallback (lower capacity)
//   stepfun-ai/step-3.5-flash              → ✅  8.8s  — too slow
//   microsoft/phi-4-mini-instruct           → ❌  35s+  — timeout on free tier
//   nvidia/mistral-nemo-minitron-8b-8k-instruct → ❌  404 — not on this account tier
//   nv-mistralai/mistral-nemo-12b-instruct → ❌  404  — not on this account tier
//   writer/palmyra-creative-122b           → ❌  404  — not on this account tier
//
// Override via env var: NVIDIA_MODEL=meta/llama-3.2-3b-instruct
const NVIDIA_DEFAULT_MODEL = "openai/gpt-oss-20b";

const NVIDIA_SYSTEM_PROMPT =
  "You are a precise text rewriter. " +
  "You rewrite text so it sounds naturally human-authored, " +
  "without adding new content or changing the meaning. " +
  "Output ONLY the rewritten text — no preamble, no commentary, no explanation.";

function getNvidiaParaphrasePrompt(text: string): string {
  return (
    "Rewrite the following text in your own words.\n" +
    "Rules:\n" +
    "- Keep the exact same meaning and all factual claims.\n" +
    "- Keep approximately the same length (±10%).\n" +
    "- Use natural, varied sentence structures — mix short and long sentences.\n" +
    "- Do NOT add new information, opinions, or examples.\n" +
    "- Do NOT use bullet lists or headers that aren't already in the original.\n" +
    "- Output ONLY the rewritten text.\n\n" +
    "TEXT:\n" +
    text
  );
}

async function callNvidia(text: string): Promise<string> {
  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) throw new Error("NVIDIA_API_KEY not set");

  const model = process.env.NVIDIA_MODEL ?? NVIDIA_DEFAULT_MODEL;

  const fetchCall = fetch(`${NVIDIA_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: NVIDIA_SYSTEM_PROMPT },
        { role: "user",   content: getNvidiaParaphrasePrompt(text) },
      ],
      temperature: 0.65,
      top_p: 0.90,
      max_tokens: 4096,
    }),
  });

  const response = await withTimeout(fetchCall, NVIDIA_TIMEOUT_MS, `NVIDIA (${model})`);

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`NVIDIA API ${response.status}: ${body.slice(0, 200)}`);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any = await response.json();
  const out: string | undefined = data?.choices?.[0]?.message?.content?.trim();
  if (!out) throw new Error("NVIDIA returned empty response");
  return out;
}

// Processes the full merged text through NVIDIA NIM.
// Batches into groups of 3 paragraphs to stay well within token limits.
// Skips silently (returns original text) if NVIDIA_API_KEY is absent or any
// unrecoverable error occurs, so fast mode is never blocked by this pass.
async function nvidiaFingerprintBreakPass(text: string): Promise<string> {
  if (!process.env.NVIDIA_API_KEY) return text;

  const paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim().length > 0);

  // Short texts — single block call for better coherence.
  if (paragraphs.length <= 2) {
    try {
      const result = await callNvidia(text);
      return validateChunkOutput(result, text);
    } catch (err) {
      console.warn(
        "[nvidia] single-block pass failed, skipping:",
        err instanceof Error ? err.message : err
      );
      return text;
    }
  }

  // Longer texts — batch in groups of 3 paragraphs, run in parallel.
  const batches: string[] = [];
  for (let i = 0; i < paragraphs.length; i += 3) {
    batches.push(paragraphs.slice(i, i + 3).join("\n\n"));
  }

  const results = await Promise.all(
    batches.map((batch) =>
      callNvidia(batch)
        .then((r) => validateChunkOutput(r, batch))
        .catch((err) => {
          console.warn(
            "[nvidia] batch failed, using original:",
            err instanceof Error ? err.message : err
          );
          return batch;
        })
    )
  );

  return results.join("\n\n");
}

// ─── Pipeline Passes ──────────────────────────────────────────────────────────

function validateChunkOutput(output: string, fallback: string): string {
  return output.trim().length > 0 ? output.trim() : fallback;
}

function getFastSinglePassPrompt(text: string, contentType: ContentType): string {
  const isFormal = contentType === "essay" || contentType === "academic";
  const registerNote = isFormal
    ? "Formal academic voice. No contractions. Plain academic vocabulary."
    : "Natural human voice. Plain everyday vocabulary. Contractions OK.";

  return `Rewrite this text so it reads like a real human wrote it.
Use plain, common words. Preserve the meaning exactly.

${registerNote}

Do:
- Remove ornate vocabulary (utilize→use, leverage→use, facilitate→help,
  multifaceted→complex, pervasive→common, permeate→reach)
- Remove signpost phrases ("furthermore", "moreover", "in conclusion",
  "it is important to note")
- Change a repeated sentence opener if two in a row start the same

Do NOT:
- Add parenthetical asides or em-dash asides
- Add "honestly", "frankly", "in practice" or similar opinion markers
- Expand the text — keep it 90-105% of input length
- Replace a plain word with a fancier synonym

Output only the rewritten text. No preamble.

TEXT:
${text}`;
}

async function singlePassHumanize(
  text: string,
  contentType: ContentType
): Promise<string> {
  try {
    const output = await callModel(getFastSinglePassPrompt(text, contentType), SINGLE_PASS_SETTINGS);
    return validateChunkOutput(output, text);
  } catch {
    return text;
  }
}

// Pass 1 — Structural rewrite per chunk (parallel).
async function structuralPass(
  chunks: string[],
  contentType: ContentType
): Promise<string[]> {
  return Promise.all(
    chunks.map((chunk, i) =>
      callModel(getStructuralPrompt(chunk, contentType, i), STRUCTURAL_SETTINGS)
        .then(o => validateChunkOutput(o, chunk))
        .catch(() => chunk)
    )
  );
}

// Pass 2 — Semantic naturalness per chunk (parallel).
async function semanticPass(
  chunks: string[],
  contentType: ContentType
): Promise<string[]> {
  return Promise.all(
    chunks.map((chunk, i) =>
      callModel(getSemanticPrompt(chunk, contentType, i), SEMANTIC_SETTINGS)
        .then(o => validateChunkOutput(o, chunk))
        .catch(() => chunk)
    )
  );
}

// Pass 3 — Selective mutation on full merged text. Uses stronger model.
async function mutationPass(
  text: string,
  contentType: ContentType
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");

  const { GoogleGenerativeAI } = await import("@google/generative-ai");
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: MUTATION_MODEL,
    generationConfig: {
      temperature: MUTATION_SETTINGS.temperature,
      topP: MUTATION_SETTINGS.topP,
      maxOutputTokens: MUTATION_SETTINGS.maxOutputTokens,
    },
  });
  const result = await withTimeout(
    model.generateContent(getMutationPrompt(text, contentType)),
    GEMINI_TIMEOUT_MS,
    "Gemini mutation call"
  );
  return result.response.text().trim() || text;
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

export function antiPatternPass(text: string, register: SourceRegister): string {
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

export function rhetoricalSuppressionPass(text: string): string {
  let result = text;
  for (const [pattern, replacement] of RHETORICAL_REPLACEMENTS) {
    result = result.replace(pattern, replacement);
  }
  return result;
}

// ─── Pattern Killers ─────────────────────────────────────────────────────────
// Targets specific LLM fingerprints observed in detector flags:
//   - "X, Y, and Z alike" triplet endings (ZeroGPT/QuillBot magnet)
//   - "The evidence is clear." / "It is clear that" dead phrases
//   - "is one of X's biggest/most Y" topic-sentence template
//   - Tautological adjective pairs ("careful and thoughtful")
// These all survived the old antiPatternPass because that pass only touches
// sentence openers. These run as replacements anywhere in the paragraph.

const TAUTOLOGY_PAIRS: Array<[RegExp, string]> = [
  [/\bcareful and thoughtful\b/gi, "careful"],
  [/\bthoughtful and careful\b/gi, "thoughtful"],
  [/\bfair and equitable\b/gi, "fair"],
  [/\bequitable and fair\b/gi, "equitable"],
  [/\beffective and engaging\b/gi, "engaging"],
  [/\bengaging and effective\b/gi, "effective"],
  [/\bmeaningful and lasting\b/gi, "lasting"],
  [/\blasting and meaningful\b/gi, "lasting"],
  [/\bclear and concise\b/gi, "clear"],
  [/\bconcise and clear\b/gi, "concise"],
  [/\bsafe and secure\b/gi, "secure"],
  [/\bsimple and easy\b/gi, "simple"],
  [/\beasy and simple\b/gi, "easy"],
  [/\bthorough and comprehensive\b/gi, "thorough"],
  [/\bcomprehensive and thorough\b/gi, "comprehensive"],
  [/\bcritical and measured\b/gi, "measured"],
  [/\bmeasured and critical\b/gi, "measured"],
  [/\brigorous and systematic\b/gi, "rigorous"],
  [/\bethical and responsible\b/gi, "responsible"],
  [/\bresponsible and ethical\b/gi, "responsible"],
  [/\binnovative and creative\b/gi, "creative"],
  [/\bcreative and innovative\b/gi, "creative"],
];

const DEAD_PHRASES: RegExp[] = [
  /\s*\bThe evidence is clear\.\s*/gi,
  /\s*\bIt is clear that\b\s*/gi,
  /\s*\bThis cannot be overstated\.\s*/gi,
  /\s*\bThis is especially true\b\s*/gi,
  /\s*\bneedless to say,?\s*/gi,
  // Legacy lagom-humanizer bridge phrases — defensive strip in case an
  // older burstinessInjector output is still cached or re-entered.
  /\s*\bThis distinction matters\.\s*/gi,
  /\s*\bThat much is certain\.\s*/gi,
  /\s*\bThe pattern holds\.\s*/gi,
  /\s*\bThis warrants scrutiny\.\s*/gi,
  /\s*\bThe implications run deep\.\s*/gi,
  /\s*\bNot all scholars agree\.\s*/gi,
  /\s*\bThe data confirms this\.\s*/gi,
  /\s*\bThis remains contested\.\s*/gi,
  /\s*\bThat part matters\.\s*/gi,
  /\s*\bThis changed things\.\s*/gi,
  /\s*\bThe difference is real\.\s*/gi,
  /\s*\bNot everyone agrees\.\s*/gi,
  /\s*\bSimple as that\.\s*/gi,
  /\s*\bThe point stands\.\s*/gi,
];

// Legacy lagom-humanizer parentheticals — defensive strip.
const LEGACY_PARENS: RegExp[] = [
  /\s*\(though this remains debated\)/gi,
  /\s*\(at least in the current literature\)/gi,
  /\s*\(admittedly\)/gi,
  /\s*\(to varying degrees\)/gi,
  /\s*\(with some exceptions\)/gi,
  /\s*\(broadly speaking\)/gi,
  /\s*\(in most formulations\)/gi,
  /\s*\(as one might expect\)/gi,
  /\s*\(though not always\)/gi,
  /\s*\(at least in theory\)/gi,
  /\s*\(more or less\)/gi,
  /\s*\(to some extent\)/gi,
  /\s*\(roughly speaking\)/gi,
  /\s*\(in most cases\)/gi,
];

export function patternKillerPass(text: string): string {
  let result = text;

  // 1. Triplet ending with "alike" — strip "alike"
  //    "teachers, rule-makers, and students alike." → "teachers, rule-makers, and students."
  result = result.replace(/(\b\w+),\s+([\w-]+),\s+and\s+([\w-]+)\s+alike\b/gi, "$1, $2, and $3");

  // 2. "X and Y alike" — 2-term version
  result = result.replace(/(\b[\w-]+)\s+and\s+([\w-]+)\s+alike\b/gi, "$1 and $2");

  // 3. Dead topic-sentence phrases
  for (const re of DEAD_PHRASES) {
    result = result.replace(re, " ");
  }

  // 3b. Legacy parentheticals from older structuralPerplexityInjector
  for (const re of LEGACY_PARENS) {
    result = result.replace(re, "");
  }

  // 3c. Em-dash list fingerprint: "X — along with Y" → "X, and Y"
  result = result.replace(/\s+—\s+along\s+with\s+/gi, ", and ");

  // 3d. "However," at sentence start → drop. Weak contrast signal that
  // reads as textbook topic-sentence cadence.
  result = result.replace(/(^|\.\s+|\?\s+|!\s+|\n)However,\s+/g, "$1");

  // 4. Topic-sentence template: "is one of X's biggest/largest/greatest/most N"
  //    "is one of AI's biggest benefits" → "is a top AI benefit"
  result = result.replace(
    /\bis\s+one\s+of\s+([\w'-]+?)(?:'s|'s)\s+(biggest|largest|greatest|most\s+important|most\s+significant|key)\s+(\w+s)\b/gi,
    (_m, owner: string, _mag: string, noun: string) => `is a top ${owner} ${noun.replace(/s$/, "")}`
  );

  // 5. Generic "is one of the biggest/most N" → "is a major N"
  result = result.replace(
    /\bis\s+one\s+of\s+the\s+(biggest|largest|greatest|most\s+important|most\s+significant)\s+(\w+s)\b/gi,
    (_m, _mag: string, noun: string) => `is a major ${noun.replace(/s$/, "")}`
  );

  // 6. Tautological adjective pairs
  for (const [pattern, replacement] of TAUTOLOGY_PAIRS) {
    result = result.replace(pattern, replacement);
  }

  // 7. Collapse any double-spaces introduced by the kills
  result = result.replace(/[ \t]{2,}/g, " ").replace(/\s+([.!?,;:])/g, "$1");

  return result;
}

// ─── Plain Language Pass (formerly Perplexity Injector) ─────────────────────
// INVERTED from the old design. Modern AI detectors (GPTZero 4.4b "Possible AI
// Paraphrasing") flag ornate vocabulary as an AI-paraphrased fingerprint.
// Instead of swapping plain→fancy, we now swap fancy→plain. This destroys
// the thesaurus-attack pattern LLMs love to generate.
//
// The exported name `perplexityInjector` is kept for import compatibility
// with lib/deep-humanizer.ts.

// Each entry: fancy word/phrase → plain replacements.
// Order matters: multi-word phrases first so they match before single words.
const PLAIN_LANGUAGE_SWAPS: Array<[RegExp, string[]]> = [
  // Multi-word purple prose — highest priority
  [/\bcommands?\s+a\s+pervasive\s+authority\b/gi, ["dominates", "controls"]],
  [/\bfurnish(?:es|ed|ing)?\s+(?:professionals?\s+)?with\b/gi, ["gives", "offers"]],
  [/\ba\s+labyrinthine\s+array\s+of\b/gi, ["many", "a set of"]],
  [/\bmultifaceted\s+tapestry\s+of\b/gi, ["a mix of", "many"]],
  [/\bmeteoric\s+(?:progression|rise|growth)\b/gi, ["rapid growth", "fast growth"]],
  [/\bparadigm[- ]shifting\s+leap\s+forward\b/gi, ["big change", "breakthrough"]],
  [/\bprofoundly\s+synergistic\s+posture\b/gi, ["joint effort", "shared approach"]],
  [/\bunyielding\s+data\s+privacy\b/gi, ["strict data privacy", "firm privacy"]],
  [/\bstartling\s+efficiency\b/gi, ["speed", "efficiency"]],
  [/\bcolossal\s+data\s+repositories?\b/gi, ["huge datasets", "large databases"]],
  [/\binterrogate\s+(?:colossal\s+)?data\b/gi, ["search data", "scan data"]],
  [/\bcellular\s+scrutiny\s+of\b/gi, ["close look at", "study of"]],
  [/\bpervasive\s+authority\s+over\b/gi, ["strong impact on", "major role in"]],
  [/\bdisproportionate\s+harm\s+upon\b/gi, ["unfair harm to", "extra harm to"]],
  [/\bmarginalized\s+communities\b/gi, ["marginalized groups", "at-risk groups"]],
  [/\bincontestably\s+broad\b/gi, ["clearly wide", "unquestionably wide"]],
  [/\bunalterably\s+chart\b/gi, ["shape", "drive"]],
  [/\bsophisticated\s+outputs?\b/gi, ["results", "outputs"]],
  [/\bfoundational\s+theories?\b/gi, ["core ideas", "main ideas"]],
  [/\badministrative\s+burden\b/gi, ["paperwork load", "admin work"]],
  [/\bpredictive\s+analytics\s+platforms?\b/gi, ["predictive tools", "prediction systems"]],

  // Single-word AI-favorite verbs
  [/\binterrogate(d|s|ing)?\b/gi, ["examine", "check", "look at"]],
  [/\bemploy(s|ed|ing)?\b/gi, ["use", "use", "using"]],
  [/\butiliz(e|es|ed|ing)\b/gi, ["use", "uses", "used", "using"]],
  [/\bleverag(e|es|ed|ing)\b/gi, ["use", "uses", "used", "using"]],
  [/\bfurnish(es|ed|ing)?\b/gi, ["give", "gives", "gave", "giving"]],
  [/\bbolster(s|ed|ing)?\b/gi, ["boost", "boosts", "boosted", "boosting"]],
  [/\bameliorat(e|es|ed|ing)\b/gi, ["improve", "improves", "improved", "improving"]],
  [/\bengender(s|ed|ing)?\b/gi, ["cause", "causes", "caused", "causing"]],
  [/\belucidat(e|es|ed|ing)\b/gi, ["explain", "explains", "explained", "explaining"]],
  [/\battenuat(e|es|ed|ing)\b/gi, ["reduce", "reduces", "reduced", "reducing"]],
  [/\bcurtail(s|ed|ing)?\b/gi, ["cut", "cuts", "cut", "cutting"]],
  [/\bculminat(e|es|ed|ing)\b/gi, ["end", "ends", "ended", "ending"]],
  [/\bunderscore(s|d)?\b/gi, ["highlight", "highlights", "highlighted"]],
  [/\bforeground(s|ed|ing)?\b/gi, ["highlight", "highlights", "highlighted", "highlighting"]],
  [/\bsubstantiat(e|es|ed|ing)\b/gi, ["back up", "backs up", "backed up", "backing up"]],
  [/\bcorroborat(e|es|ed|ing)\b/gi, ["confirm", "confirms", "confirmed", "confirming"]],
  [/\bprobe(s|d)?\b/gi, ["examine", "examines", "examined"]],
  [/\bunpack(s|ed|ing)?\b/gi, ["explain", "explains", "explained", "explaining"]],
  [/\bgrapple(s|d)?\s+with\b/gi, ["deal with", "deals with", "dealt with"]],
  [/\bcontend(s|ed|ing)?\s+with\b/gi, ["face", "faces", "faced", "facing"]],

  // Adjectives / adverbs
  [/\bpervasive(ly)?\b/gi, ["widespread", "widely"]],
  [/\bcolossal\b/gi, ["huge", "massive"]],
  [/\blabyrinthine\b/gi, ["complex", "tangled"]],
  [/\bunceasing(ly)?\b/gi, ["constant", "constantly"]],
  [/\bmeteoric\b/gi, ["rapid", "fast"]],
  [/\binsidious(ly)?\b/gi, ["subtle", "subtly", "hidden"]],
  [/\bunyielding(ly)?\b/gi, ["firm", "firmly", "strict"]],
  [/\bsynergistic\b/gi, ["joint", "shared"]],
  [/\bparadigm(atic)?\b/gi, ["model", "standard"]],
  [/\bmultifaceted\b/gi, ["complex", "many-sided"]],
  [/\bquintessential(ly)?\b/gi, ["classic", "typical"]],
  [/\bparamount\b/gi, ["critical", "key"]],
  [/\bdiscernible\b/gi, ["clear", "visible"]],
  [/\bstartling(ly)?\b/gi, ["striking", "surprising"]],
  [/\bsophisticated\b/gi, ["advanced", "refined"]],
  [/\bprofound(ly)?\b/gi, ["deep", "deeply"]],
  [/\bincontestabl(e|y)\b/gi, ["clear", "clearly"]],
  [/\bunalterabl(e|y)\b/gi, ["permanent", "permanently"]],
  [/\bjudicious(ly)?\b/gi, ["careful", "carefully"]],
  [/\bmaterial(ly)?\b/gi, ["real", "clearly"]],  // "materially augment" → "clearly augment" → plain
  [/\bpivotal\b/gi, ["key", "central"]],
  [/\bconsequential\b/gi, ["important", "major"]],
  [/\bweighty\b/gi, ["serious", "heavy"]],
  [/\boutsized\b/gi, ["large", "big"]],
  [/\bsizeable\b/gi, ["large", "big"]],
  [/\bmarked(ly)?\b/gi, ["clear", "clearly"]],
  [/\bconsiderabl(e|y)\b/gi, ["notable", "notably"]],
  [/\bimmensely\b/gi, ["hugely", "very"]],
  [/\babatin[g]?\b/gi, ["slowing", "easing"]],
  [/\bnuanced\b/gi, ["subtle", "fine"]],

  // Nouns
  [/\btapestry\s+of\b/gi, ["mix of", "set of"]],
  [/\bconstellation\s+of\b/gi, ["group of", "set of"]],
  [/\bpanoply\s+of\b/gi, ["wide range of", "set of"]],
  [/\bplethora\s+of\b/gi, ["lots of", "many"]],
  [/\bmyriad\b/gi, ["many", "countless"]],
  [/\bpuzzle\b/gi, ["question", "problem"]],
  [/\bquandary\b/gi, ["dilemma", "problem"]],
  [/\bavenue\b/gi, ["path", "option"]],
  [/\borientation\s+toward\b/gi, ["approach to"]],
  [/\btack\b(?!\s+on)/gi, ["approach", "method"]],
];

export function perplexityInjector(text: string, _register: SourceRegister): string {
  void _register;
  let result = text;
  for (const [pattern, replacements] of PLAIN_LANGUAGE_SWAPS) {
    let i = 0;
    result = result.replace(pattern, (match) => {
      const rep = replacements[i % replacements.length];
      i++;
      // Preserve capitalization of the first character
      return match[0] === match[0].toUpperCase() && match[0] !== match[0].toLowerCase()
        ? rep[0].toUpperCase() + rep.slice(1)
        : rep;
    });
  }
  return result;
}

// ─── ZeroGPT N-gram Breaker ──────────────────────────────────────────────────
// Replaces 30 common AI n-gram patterns that ZeroGPT is calibrated against.

const ZEROGPT_NGRAMS: Array<[RegExp, string]> = [
  [/\bin order to\b/gi, "to"],
  [/\bas a result[,\s]/gi, "this led to "],
  [/\bas a result of\b/gi, "because of"],
  [/\bin the context of\b/gi, "within"],
  [/\ba wide range of\b/gi, "various"],
  [/\ba wide variety of\b/gi, "many kinds of"],
  [/\bhas been shown to\b/gi, "appears to"],
  [/\bhave been shown to\b/gi, "appear to"],
  [/\bthere is a need to\b/gi, "we need to"],
  [/\bthere is a need for\b/gi, "we need"],
  [/\bin addition to\b/gi, "beyond"],
  [/\bdue to the fact that\b/gi, "because"],
  [/\bwith respect to\b/gi, "regarding"],
  [/\bin terms of\b/gi, "for"],
  [/\bat the same time[,]?\b/gi, "simultaneously"],
  [/\bon the other hand[,]?\b/gi, "by contrast,"],
  [/\bin recent years\b/gi, "recently"],
  [/\bit is worth noting that\b/gi, "notably,"],
  [/\bplays a significant role in\b/gi, "significantly affects"],
  [/\bplays an important role in\b/gi, "significantly shapes"],
  [/\bis designed to\b/gi, "aims to"],
  [/\bit is important to\b/gi, "one must"],
  [/\bin this regard[,]?\b/gi, "here,"],
  [/\bto a certain extent\b/gi, "to some degree"],
  [/\ba number of\b/gi, "several"],
  [/\bon a regular basis\b/gi, "regularly"],
  [/\bin the near future\b/gi, "soon"],
  [/\btake into account\b/gi, "consider"],
  [/\btaking into account\b/gi, "considering"],
  [/\bserves as a\b/gi, "functions as a"],
];

export function zeroGPTNgramBreaker(text: string): string {
  let result = text;
  for (const [pattern, replacement] of ZEROGPT_NGRAMS) {
    result = result.replace(pattern, replacement);
  }
  return result;
}

// ─── Structural Perplexity Injector ──────────────────────────────────────────
// Injects structural surprises: parentheticals, em-dash list breaking,
// semicolon splicing. Creates token-level unpredictability ZeroGPT can't predict.

const FORMAL_PARENS = [
  "(though this remains debated)", "(at least in the current literature)",
  "(admittedly)", "(to varying degrees)", "(with some exceptions)",
  "(broadly speaking)", "(in most formulations)", "(as one might expect)",
];
const CASUAL_PARENS = [
  "(though not always)", "(at least in theory)", "(more or less)",
  "(admittedly)", "(to some extent)", "(roughly speaking)", "(in most cases)",
];
const LIST_RE = /(\b\w+),\s+(\w+),?\s+and\s+(\w+)\b/;

// Grammar-safe clause-boundary detector.
// Given a sentence, returns comma positions where the following segment
// starts with a subordinator/conjunction — these are safe insertion points
// because the text on each side is guaranteed to be a complete-ish clause.
const CLAUSE_BOUNDARY_RE = /,\s+(which|who|where|when|while|although|though|because|since|as|and|but|or|yet|so|whereas|if|unless|until|before|after)\b/i;

function findClauseBoundaryCommas(sentence: string): number[] {
  const out: number[] = [];
  const re = /,\s+(which|who|where|when|while|although|though|because|since|as|and|but|or|yet|so|whereas|if|unless|until|before|after)\b/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(sentence)) !== null) {
    out.push(m.index); // index of the comma
  }
  return out;
}

export function structuralPerplexityInjector(text: string, register: SourceRegister): string {
  // DISABLED Modes A (parenthetical insertion) and B (em-dash list break):
  // both were adding identifiable lagom-humanizer fingerprints that detectors
  // now flag. Examples observed in the wild:
  //   A → "(admittedly)", "(broadly speaking)", "(with some exceptions)"
  //   B → "fair and effective — along with truly engaging for everyone"
  // Only Mode C (comma → semicolon at clause boundary) survives — semicolons
  // are legitimate human punctuation and haven't been flagged.
  void register; void FORMAL_PARENS; void CASUAL_PARENS; void LIST_RE;
  const paragraphs = text.split(/\n\s*\n/);
  let totalMods = 0;

  const result = paragraphs.map((para) => {
    if (totalMods >= 4) return para;
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;
    let paraModCount = 0;

    const processed = sentences.map((sentence) => {
      if (paraModCount >= 1) return sentence;
      const words = sentence.split(/\s+/).length;

      // C: Comma → semicolon ONLY at a clause boundary comma.
      if (words >= 20) {
        const boundaries = findClauseBoundaryCommas(sentence);
        const candidate = boundaries.find(b => b > 15 && b < sentence.length - 15);
        if (candidate !== undefined) {
          paraModCount++; totalMods++;
          return `${sentence.slice(0, candidate)};${sentence.slice(candidate + 1)}`;
        }
      }

      return sentence;
    });

    return processed.join(" ");
  });

  return result.join("\n\n");
}
// Silence unused-symbol warning; the regex constant is retained for future
// callers but the per-call function findClauseBoundaryCommas is what runs.
void CLAUSE_BOUNDARY_RE;

// ─── Inter-Paragraph Divergence ──────────────────────────────────────────────
// Swaps a repeated content word in adjacent paragraphs with a lateral synonym.
// Targets Originality.ai's topic-uniformity signal.

const DIVERGENCE_CLUSTERS: Record<string, string[]> = {
  "approach": ["method","strategy","technique"], "method": ["approach","framework","procedure"],
  "process": ["procedure","workflow","mechanism"], "system": ["framework","structure","model"],
  "model": ["framework","paradigm","structure"], "result": ["outcome","finding","effect"],
  "outcome": ["result","consequence","impact"], "impact": ["effect","influence","consequence"],
  "role": ["function","purpose","contribution"], "factor": ["element","variable","component"],
  "aspect": ["dimension","element","feature"], "issue": ["challenge","problem","concern"],
  "challenge": ["difficulty","obstacle","issue"], "solution": ["answer","remedy","resolution"],
  "context": ["setting","environment","situation"], "data": ["evidence","information","findings"],
  "analysis": ["examination","assessment","evaluation"], "research": ["study","investigation","work"],
  "study": ["research","investigation","analysis"], "level": ["degree","extent","rate"],
  "case": ["instance","example","situation"], "example": ["case","instance","illustration"],
  "area": ["domain","field","sector"], "field": ["area","domain","discipline"],
};

const DIV_STOP = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by",
  "from","is","are","was","were","be","been","has","have","had","do","does","did","will","would",
  "could","should","may","might","this","that","these","those","it","its","they","them","their",
  "we","our","you","your","he","she","his","her","as","if","so","not","also","can","into"]);

export function interParagraphDivergencePass(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  if (paragraphs.length < 2) return text;
  const paraWords = paragraphs.map(para =>
    new Set(para.toLowerCase().replace(/[^a-z\s]/g,"").split(/\s+/).filter(w => w.length > 3 && !DIV_STOP.has(w)))
  );
  const result = paragraphs.map((para, i) => {
    if (i === 0) return para;
    const prevWords = paraWords[i - 1];
    const words = para.split(/\s+/);
    for (let j = 0; j < words.length; j++) {
      const clean = words[j].toLowerCase().replace(/[^a-z]/g, "");
      if (clean.length < 4 || DIV_STOP.has(clean)) continue;
      const alts = DIVERGENCE_CLUSTERS[clean];
      if (!alts || !prevWords.has(clean)) continue;
      const available = alts.filter(a => !paraWords[i].has(a));
      if (available.length === 0) continue;
      const replacement = available[(i + j) % available.length];
      const punct = words[j].slice(clean.length);
      const isCap = words[j][0] === words[j][0].toUpperCase() && words[j][0] !== words[j][0].toLowerCase();
      const replaced = (isCap ? replacement[0].toUpperCase() + replacement.slice(1) : replacement) + punct;
      return [...words.slice(0, j), replaced, ...words.slice(j + 1)].join(" ");
    }
    return para;
  });
  return result.join("\n\n");
}

// ─── GPTZero Burstiness Injector ─────────────────────────────────────────────
// Measures sentence-length std dev per paragraph. If < 5, forces variance by
// splitting the longest sentence or inserting a short register-safe bridge.

const FORMAL_BRIDGES = [
  "This distinction matters.", "The evidence is clear.", "That much is certain.",
  "The pattern holds.", "This warrants scrutiny.", "The implications run deep.",
  "Not all scholars agree.", "The data confirms this.", "This remains contested.",
];
const CASUAL_BRIDGES = [
  "That part matters.", "This changed things.", "It shows.", "The difference is real.",
  "Not everyone agrees.", "Simple as that.", "Worth noting.", "The point stands.",
];

export function burstinessInjector(text: string, register: SourceRegister): string {
  const isFormal = register === "academic" || register === "formal";
  const bridges = isFormal ? FORMAL_BRIDGES : CASUAL_BRIDGES;
  const paragraphs = text.split(/\n\s*\n/);
  let totalInjections = 0;

  const result = paragraphs.map((para, pIdx) => {
    if (totalInjections >= 4) return para;
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;
    const lengths = sentences.map(s => s.split(/\s+/).length);
    const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    const stdDev = Math.sqrt(lengths.reduce((sum, l) => sum + Math.pow(l - avg, 2), 0) / lengths.length);

    // Fire if stdDev < 8 (was 5 — now catches nearly-uniform paragraphs too)
    if (stdDev >= 8) return para;

    const modified = [...sentences];
    let didModify = false;

    // Strategy A: Split longest sentence ONLY at a clause-boundary comma
    // (one followed by a coordinating conjunction). Splitting at an arbitrary
    // comma produces sentence fragments like "Modern healthcare's field."
    const longestIdx = lengths.indexOf(Math.max(...lengths));
    if (lengths[longestIdx] > 20) {
      const target = modified[longestIdx];
      // Match ", and|but|or|yet|so " — coordinating conjunctions that can
      // become the start of a new sentence after the comma is promoted.
      const coordRe = /,\s+(and|but|or|yet|so)\s+/i;
      const m = coordRe.exec(target);
      if (m && m.index > 20 && m.index < target.length - 20) {
        const commaPos = m.index;
        const afterCommaStart = commaPos + m[0].length;
        const first = target.slice(0, commaPos).trimEnd() + ".";
        const rest = target.slice(afterCommaStart).trimStart();
        // Capitalize first letter of the new sentence.
        modified[longestIdx] = first;
        modified.splice(longestIdx + 1, 0, rest.charAt(0).toUpperCase() + rest.slice(1));
        didModify = true;
        totalInjections++;
      }
    }

    // Strategy B DISABLED: bridge injection was inserting canned phrases
    // like "This distinction matters." and "The evidence is clear." that
    // detectors now flag as lagom-humanizer fingerprints. Strategy A above
    // (sentence split at coordinator) handles burstiness without injecting
    // canned text.
    void bridges; void pIdx;

    return modified.join(" ");
  });

  return result.join("\n\n");
}

// ─── Em-Dash Injector ────────────────────────────────────────────────────────
// Em-dashes are strongly human. GPTZero's model rarely sees them in AI output
// → injecting them raises per-token perplexity and burstiness simultaneously.
// Targets sentences 20+ words with a natural mid-point for insertion.

const FORMAL_EM_ASIDES = [
  "and this distinction matters",
  "at least in most formulations",
  "though the evidence is mixed",
  "a point often overlooked",
  "the data bears this out",
  "not a trivial consideration",
  "worth noting here",
  "broadly speaking",
];
const CASUAL_EM_ASIDES = [
  "and that's the key part",
  "though not always",
  "at least in theory",
  "which matters a lot",
  "worth flagging",
  "more or less",
];

function emDashInjector(text: string, register: SourceRegister): string {
  const isFormal = register === "academic" || register === "formal";
  const asides = isFormal ? FORMAL_EM_ASIDES : CASUAL_EM_ASIDES;
  const paragraphs = text.split(/\n\s*\n/);
  let totalInjections = 0;

  const result = paragraphs.map((para, pIdx) => {
    if (totalInjections >= 2) return para;
    const sentences = getSentences(para);
    const modified = sentences.map((sentence, sIdx) => {
      if (totalInjections >= 2) return sentence;
      const words = sentence.split(/\s+/).length;
      if (words < 22) return sentence;
      // Only inject into every 2nd qualifying sentence to avoid over-formatting
      if ((pIdx + sIdx) % 2 !== 0) return sentence;

      // Grammar-safe: only insert em-dash asides at an existing comma that
      // sits at a clause boundary (followed by a subordinator/conjunction).
      // This guarantees the aside lands between complete clauses and never
      // inside a compound noun like "artificial intelligence".
      const boundaries = findClauseBoundaryCommas(sentence);
      const commaPos = boundaries.find(b => b > 15 && b < sentence.length - 20);
      if (commaPos === undefined) return sentence;

      const aside = asides[(pIdx + sIdx + totalInjections) % asides.length];
      const before = sentence.slice(0, commaPos); // drop the comma
      const after = sentence.slice(commaPos + 1).trimStart();
      totalInjections++;
      return `${before}—${aside}—${after}`;
    });
    return modified.join(" ");
  });

  return result.join("\n\n");
}

// ─── Opener Diversity Pass ────────────────────────────────────────────────────
// Detects duplicate sentence-opening words within a paragraph, swaps the
// second occurrence from a curated prefix table. Max 2 swaps per paragraph.

const OPENER_SWAPS: Record<string, string[]> = {
  "this": ["That","The","It"], "the": ["Its","That","Each"],
  "it": ["This","That","The result"], "these": ["Such","Those","The"],
  "there": ["Here","That said,","In practice,"], "that": ["This","It","The fact"],
  "they": ["Both","Each of them","Those"], "we": ["Our","In doing so,","The team"],
  "one": ["A key","Each","An important"], "when": ["As","Once","Wherever"],
  "while": ["Although","Even as","As"], "since": ["Because","As","Given that"],
  "for": ["To","In order to","As a way to"], "with": ["Using","By","Through"],
  "by": ["Through","Via","Using"], "in": ["Within","Across","At"],
  "as": ["While","When","Given that"], "if": ["Should","When","Assuming"],
  "most": ["Many","The majority of","A large portion of"],
  "many": ["Several","A number of","Numerous"],
  "some": ["A few","Certain","Several"], "each": ["Every","Any given","Individual"],
  "both": ["Each","Either","The two"],
};

// Demonstrative-class openers — "This", "These", "Those", "That", "It" all
// act as pointers-to-previous-sentence and cluster in LLM output. Detectors
// key heavily on this cadence, so we treat them as a single class and force
// a swap on any occurrence past the first in a paragraph.
const DEMONSTRATIVE_OPENERS = new Set(["this", "these", "those", "that", "it"]);

export function openerDiversityPass(text: string): string {
  return text.split(/\n\s*\n/).map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;
    let swaps = 0;
    let demonstrativeCount = 0;
    const seen = new Map<string, number>();
    const rewritten = sentences.map((s, i) => {
      if (swaps >= 2) return s;
      const firstWord = s.split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, "");

      // Demonstrative-class tracking: any second+ demonstrative opener
      // gets treated as a repeat even if the exact word differs.
      if (DEMONSTRATIVE_OPENERS.has(firstWord)) {
        demonstrativeCount++;
        if (demonstrativeCount > 1) {
          const alts = OPENER_SWAPS[firstWord];
          if (alts) {
            const body = s.slice(firstWord.length).trimStart();
            const alt = alts[(i + para.length) % alts.length];
            swaps++;
            return alt + (alt.endsWith(",") ? " " + body.charAt(0).toLowerCase() + body.slice(1) : " " + body);
          }
        }
        seen.set(firstWord, i);
        return s;
      }

      const prevIdx = seen.get(firstWord);
      if (prevIdx !== undefined && i > prevIdx) {
        const alts = OPENER_SWAPS[firstWord];
        if (alts) {
          const body = s.slice(firstWord.length).trimStart();
          const alt = alts[(i + para.length) % alts.length];
          swaps++;
          return alt + (alt.endsWith(",") ? " " + body.charAt(0).toLowerCase() + body.slice(1) : " " + body);
        }
      } else { seen.set(firstWord, i); }
      return s;
    });
    return rewritten.join(" ");
  }).join("\n\n");
}

// ─── Grammar & Duplicate Cleanup ─────────────────────────────────────────────
// Safety net that runs AFTER the deterministic injectors to catch damage they
// may have introduced:
//   1. Near-duplicate sentences within the same paragraph (bridge sentences
//      being reinserted across iterations).
//   2. Obvious sentence fragments (no verb, under 5 words, starting with a
//      lowercase word) are merged into the previous sentence.

function normalizeForDedup(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim();
}

function looksLikeFragment(s: string): boolean {
  const trimmed = s.trim();
  if (trimmed.length === 0) return true;
  const words = trimmed.split(/\s+/);
  if (words.length < 4) return true;
  // Starts with a lowercase letter → likely a continuation spliced wrong.
  const first = trimmed[0];
  if (first >= "a" && first <= "z") return true;
  // No verb-like token (very rough — any word ending in common verb suffixes
  // or a small set of auxiliaries).
  const hasVerb = /\b(is|are|was|were|be|been|being|has|have|had|do|does|did|will|would|can|could|should|may|might|must|\w+s|\w+ed|\w+ing)\b/i.test(trimmed);
  return !hasVerb;
}

export function grammarAndDedupPass(text: string): string {
  const paragraphs = text.split(/\n\s*\n/);
  const seenGlobal = new Set<string>();

  const cleaned = paragraphs.map((para) => {
    const sentences = getSentences(para);
    if (sentences.length === 0) return para;

    const kept: string[] = [];
    const seenLocal = new Set<string>();

    for (const s of sentences) {
      const norm = normalizeForDedup(s);
      if (norm.length === 0) continue;

      // Drop exact near-duplicates within the paragraph AND globally if short
      // (short sentences like "The data confirms this." are high-risk).
      if (seenLocal.has(norm)) continue;
      const wordCount = norm.split(" ").length;
      if (wordCount <= 8 && seenGlobal.has(norm)) continue;

      // Merge fragments into the previous sentence.
      if (looksLikeFragment(s) && kept.length > 0) {
        const prev = kept[kept.length - 1].replace(/[.!?]+$/, "");
        kept[kept.length - 1] = `${prev}, ${s.trim().replace(/^[.,;:\s]+/, "")}`;
        continue;
      }

      kept.push(s);
      seenLocal.add(norm);
      if (wordCount <= 8) seenGlobal.add(norm);
    }

    return kept.join(" ");
  });

  return cleaned.join("\n\n");
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

async function finishHumanizedText(
  text: string,
  register: SourceRegister,
  inputWordCount: number
): Promise<string> {
  // Glory-days deterministic stack (restored from commit 1007a90), minus
  // emDashInjector which was proven to REGRESS ZeroGPT/QuillBot scores
  // (8%→24% and 16%→62% observed during earlier testing).
  //
  // Each pass targets a specific detector signal:
  //  - antiPatternPass: kills AI paragraph openers
  //  - rhetoricalSuppressionPass: strips connector/fluency overload
  //  - zeroGPTNgramBreaker: 30 hardcoded n-gram pattern rules
  //  - structuralPerplexityInjector: parentheticals, semicolons, clause breaks
  //  - perplexityInjector: up to 3 rare synonym swaps per paragraph
  //  - interParagraphDivergencePass: cross-paragraph vocab variance
  //  - burstinessInjector: forces sentence-length variance (GPTZero #1 signal)
  //  - openerDiversityPass: no duplicate sentence-starting words
  //  - grammarAndDedupPass: post-injector cleanup (kept from recent work)
  const cleaned = antiPatternPass(text, register);
  const suppressed = rhetoricalSuppressionPass(cleaned);
  const patternKilled = patternKillerPass(suppressed);
  const ngramBroken = zeroGPTNgramBreaker(patternKilled);
  const structPerplexed = structuralPerplexityInjector(ngramBroken, register);
  const perplexed = perplexityInjector(structPerplexed, register);
  const diverged = interParagraphDivergencePass(perplexed);
  const bursty = burstinessInjector(diverged, register);
  const opened = openerDiversityPass(bursty);
  const grammarClean = grammarAndDedupPass(opened);
  const lengthCapped = enforceLengthDiscipline(grammarClean, inputWordCount);

  // Length discipline: expand iteratively until we clear the 92% floor, or
  // we've tried twice (each Gemini call costs ~5s + we don't want to spin
  // forever if the model refuses to comply).
  const MIN_ACCEPTABLE = Math.ceil(inputWordCount * 0.92);
  const TARGET = Math.round(inputWordCount * 0.98);
  const HARD_CEILING = Math.round(inputWordCount * 1.12);

  let current = lengthCapped;
  let currentWords = current.split(/\s+/).length;

  for (let attempt = 0; attempt < 2 && currentWords < MIN_ACCEPTABLE; attempt++) {
    const deficit = inputWordCount - currentWords;
    const expansionPrompt = `Expand this text to approximately ${TARGET} words (currently ${currentWords} words — add about ${deficit} more words of relevant elaboration, examples, and concrete detail WITHIN the existing sentences and paragraphs). Do NOT add new sections or new topics. Do NOT change any facts. Preserve the exact register, voice, and style. Output ONLY the expanded text — no preamble, no commentary.

TEXT:
${current}`;
    const expanded = await callModel(expansionPrompt, { temperature: 0.70, topP: 0.90, maxOutputTokens: 4096 });
    const trimmed = expanded.trim();
    if (trimmed && trimmed.split(/\s+/).length > currentWords) {
      current = trimmed;
      currentWords = current.split(/\s+/).length;
    } else {
      // Expansion didn't help — stop trying, avoid wasting more calls.
      break;
    }
  }

  // Final length clamp: allow up to 112% (not the old 105%) so a slight
  // over-expansion isn't brutally truncated back to a short output.
  return enforceLengthDiscipline(current, Math.round(inputWordCount * (HARD_CEILING / inputWordCount)));
}

// ─── Translation Round-Trip Humanization ──────────────────────────────────────
// Deep-mode nuclear-reset weapon only. NOT used by fast-mode humanize() —
// see commit history for why (regression when used as the primary pipeline).
// Called by lib/deep-humanizer.ts pivot-rotation fallback when iterative
// sentence surgery plateaus.

export type PivotLanguage = "Chinese" | "Arabic" | "Japanese" | "Korean" | "Spanish" | "French" | "German";

// Chinese/Arabic/Japanese/Korean have fundamentally different syntax from English
// (no articles, different word order, different tense/aspect handling).
// Back-translating from these forces genuinely restructured English rather than
// the near-identical paraphrase you get from European pivots.
const TRANSLATION_TEMP = 0.4;
const BACK_TRANSLATION_TEMP = 0.7;

interface PlaceholderMap {
  [key: string]: string;
}

// Extract content that MUST NOT translate (URLs, emails, code, etc.)
// Placeholders are ASCII-safe and survive translation unchanged.
function extractPlaceholders(text: string): { stripped: string; map: PlaceholderMap } {
  const map: PlaceholderMap = {};
  let counter = 0;
  const next = (prefix: string) => `ZZ${prefix}${counter++}ZZ`;

  let result = text;

  // URLs
  result = result.replace(/https?:\/\/\S+/g, (m) => { const k = next("U"); map[k] = m; return k; });
  // Emails
  result = result.replace(/\b[\w.-]+@[\w.-]+\.\w+\b/g, (m) => { const k = next("E"); map[k] = m; return k; });
  // Fenced code blocks
  result = result.replace(/```[\s\S]*?```/g, (m) => { const k = next("C"); map[k] = m; return k; });
  // Inline code
  result = result.replace(/`[^`\n]+`/g, (m) => { const k = next("I"); map[k] = m; return k; });
  // Currency amounts
  result = result.replace(/\$\d+(?:[,.]\d+)*(?:\s*(?:million|billion|thousand|trillion|M|B|K))?/gi, (m) => { const k = next("M"); map[k] = m; return k; });
  // Percentages (keep them — translation handles OK but number formatting varies)
  result = result.replace(/\b\d+(?:\.\d+)?%/g, (m) => { const k = next("P"); map[k] = m; return k; });
  // Years (4-digit)
  result = result.replace(/\b(19|20)\d{2}\b/g, (m) => { const k = next("Y"); map[k] = m; return k; });

  return { stripped: result, map };
}

function restorePlaceholders(text: string, map: PlaceholderMap): string {
  let result = text;
  for (const [key, value] of Object.entries(map)) {
    // Replace all occurrences (translation might duplicate them, rarely)
    result = result.split(key).join(value);
  }
  return result;
}

function getRegisterInstruction(register: SourceRegister): string {
  switch (register) {
    case "academic":
      return "Formal academic English. No contractions. Plain academic vocabulary — no ornate synonyms.";
    case "formal":
      return "Formal professional English. No contractions. Plain business vocabulary.";
    case "informal":
      return "Casual conversational English. Contractions are welcome. Plain everyday words.";
    case "neutral":
    default:
      return "Natural professional English. Contractions OK where they sound natural. Plain everyday vocabulary.";
  }
}

async function translateToPivot(text: string, pivot: PivotLanguage): Promise<string> {
  const prompt = `Translate the following English text to ${pivot}.

CRITICAL RULES:
- Translate literally and accurately
- Preserve ALL numbers, names, dates, and technical terms EXACTLY
- Keep any placeholder tokens (strings like ZZU0ZZ, ZZE1ZZ, ZZP2ZZ — patterns starting and ending with ZZ) UNCHANGED — do not translate them
- Preserve paragraph breaks exactly
- Output ONLY the ${pivot} translation. No preamble, no explanation.

ENGLISH TEXT:
${text}

${pivot.toUpperCase()} TRANSLATION:`;

  return callGemini(prompt, { temperature: TRANSLATION_TEMP, topP: 0.9, maxOutputTokens: 8192 });
}

async function translateFromPivot(text: string, pivot: PivotLanguage, register: SourceRegister): Promise<string> {
  const registerInstruction = getRegisterInstruction(register);

  const prompt = `Translate the following ${pivot} text into natural, fluent, grammatically correct English.

TONE/REGISTER:
${registerInstruction}

CRITICAL RULES:
- Output MUST be 100% grammatically correct English — fix any awkward phrasing or word-order artifacts from the source language
- Vary sentence structure: some short (under 12 words), some long (over 22 words), avoid uniformity
- Use PLAIN everyday vocabulary — the simplest word that fits is always correct
- Do NOT use: "utilize", "leverage", "facilitate", "multifaceted", "pervasive", "paramount", "profound", "myriad", "plethora", "furthermore", "moreover", "additionally", "it is worth noting"
- Active voice preferred over passive where natural
- Keep any placeholder tokens (strings starting and ending with ZZ) UNCHANGED
- Preserve paragraph breaks exactly
- Do not add parenthetical asides, em-dash asides, or editorial comments
- Preserve ALL facts, numbers, names, dates, and technical terms exactly
- Output ONLY the English translation. No preamble, no commentary.

${pivot.toUpperCase()} TEXT:
${text}

ENGLISH TRANSLATION:`;

  return callGemini(prompt, { temperature: BACK_TRANSLATION_TEMP, topP: 0.9, maxOutputTokens: 8192 });
}

// Final vocabulary scrub: catch any ornate words that survived the round-trip.
// Case-preserving for sentence-start capitalization.
function plainVocabScrubber(text: string): string {
  const replacements: Array<[RegExp, string]> = [
    [/\butilize\b/g, "use"], [/\bUtilize\b/g, "Use"],
    [/\butilizes\b/g, "uses"], [/\bUtilizes\b/g, "Uses"],
    [/\butilized\b/g, "used"], [/\bUtilized\b/g, "Used"],
    [/\butilizing\b/g, "using"], [/\bUtilizing\b/g, "Using"],
    [/\bleverage\b/g, "use"], [/\bLeverage\b/g, "Use"],
    [/\bleverages\b/g, "uses"], [/\bleveraged\b/g, "used"],
    [/\bleveraging\b/g, "using"],
    [/\bfacilitate\b/g, "help"], [/\bFacilitate\b/g, "Help"],
    [/\bfacilitates\b/g, "helps"], [/\bfacilitated\b/g, "helped"],
    [/\bfacilitating\b/g, "helping"],
    [/\bmultifaceted\b/gi, "complex"],
    [/\bpervasive\b/gi, "widespread"],
    [/\bpermeate\b/gi, "reach into"],
    [/\bpermeates\b/gi, "reaches into"],
    [/\bpermeating\b/gi, "reaching into"],
    [/\bparamount\b/gi, "key"],
    [/\bprofound\b/gi, "deep"],
    [/\bProfound\b/g, "Deep"],
    [/\bintricate\b/gi, "complex"],
    [/\bmyriad\b/gi, "many"],
    [/\bplethora of\b/gi, "many"],
    [/\bdelve into\b/gi, "examine"],
    [/\bdelves into\b/gi, "examines"],
    [/\bFurthermore,?\s*/g, ""],
    [/\bfurthermore,?\s*/g, ""],
    [/\bMoreover,?\s*/g, ""],
    [/\bmoreover,?\s*/g, ""],
    [/\bIn conclusion,?\s*/gi, ""],
    [/\bIt is important to note that\s*/gi, ""],
    [/\bIt is worth noting that\s*/gi, ""],
  ];

  let result = text;
  for (const [pattern, replacement] of replacements) {
    result = result.replace(pattern, replacement);
  }

  // Cleanup: fix empty-leading-comma / double-space artifacts from removals
  result = result.replace(/\s{2,}/g, " ");
  result = result.replace(/\n /g, "\n");
  result = result.replace(/([.!?])\s+([a-z])/g, (_m, p, c) => `${p} ${c.toUpperCase()}`);
  return result.trim();
}

// Main round-trip function. Handles placeholders, translation, restoration,
// and final vocabulary cleanup. Returns the humanized text.
export async function roundTripHumanize(
  text: string,
  register: SourceRegister,
  pivot: PivotLanguage = "Chinese"
): Promise<string> {
  const { stripped, map } = extractPlaceholders(text);

  // Step 1: English → pivot language
  const pivotText = await translateToPivot(stripped, pivot);

  // Step 2: pivot language → English (register-aware)
  const backText = await translateFromPivot(pivotText, pivot, register);

  // Step 3: Restore placeholders
  const restored = restorePlaceholders(backText, map);

  // Step 4: Final plain-vocab scrub (catches any ornate words that slipped through)
  return plainVocabScrubber(restored);
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

  // Very short inputs (<30 words): single-pass is safer than multi-pass —
  // chunking overhead + meaning drift risk outweigh the quality gain.
  if (inputWordCount < 30) {
    const singlePass = await singlePassHumanize(truncated, contentType);
    return finishHumanizedText(singlePass, register, inputWordCount);
  }

  // GLORY-DAYS pipeline (restored from commit 1007a90):
  //   Chunk → structural (parallel) → semantic (parallel) → mutation (gated)
  //   → finishHumanizedText (9 deterministic passes).
  //
  // This is the pipeline that produced 99% Human on GPTZero, 0% on ZeroGPT,
  // ~0% on QuillBot — WITHOUT deep mode. Round-trip translation has been
  // demoted to a deep-mode-only weapon (see roundTripHumanize, still exported
  // for use by lib/deep-humanizer.ts nuclear-reset path).
  try {
    const chunks = splitIntoVariableChunks(truncated);

    // Pass 1: Structural rewrite (parallel per chunk)
    const structural = await structuralPass(chunks, contentType);

    // Pass 2: Semantic naturalness (parallel per chunk)
    const semantic = await semanticPass(structural, contentType);
    const merged = semantic.join("\n\n");

    // Tier 0 — NVIDIA Fingerprint Break: run merged Gemini output through a
    // completely different model family (NVIDIA NIM free API) to shatter the
    // Gemini statistical signature that detectors are trained on.
    // Silently skipped if NVIDIA_API_KEY is not configured.
    const fingerprintBroken = await nvidiaFingerprintBreakPass(merged);

    // Pass 3: Targeted mutation — only fire if internal detector still
    // thinks the text is synthetic. Skipping when already human-sounding
    // avoids over-paraphrasing and preserves meaning/tone/word count.
    const { score: midScore } = detectAI(fingerprintBroken);
    const mutated = midScore > 25
      ? await mutationPass(fingerprintBroken, contentType).catch(() => fingerprintBroken)
      : fingerprintBroken;

    return finishHumanizedText(mutated, register, inputWordCount);
  } catch (err) {
    console.error(
      "[humanize] multi-pass failed, falling back to single-pass:",
      err instanceof Error ? err.message : err
    );
    const singlePass = await singlePassHumanize(truncated, contentType);
    return finishHumanizedText(singlePass, register, inputWordCount);
  }
}
