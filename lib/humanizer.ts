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

const STRUCTURAL_SETTINGS: GenSettings = { temperature: 0.85, topP: 0.95, maxOutputTokens: 4096 };
const SEMANTIC_SETTINGS: GenSettings  = { temperature: 0.75, topP: 0.90, maxOutputTokens: 4096 };
const MUTATION_SETTINGS: GenSettings  = { temperature: 0.88, topP: 0.92, maxOutputTokens: 4096 };
const GEMINI_TIMEOUT_MS = 30_000;
const SHORT_FAST_PATH_WORDS = 120;
const SINGLE_PASS_SETTINGS: GenSettings = { temperature: 0.78, topP: 0.90, maxOutputTokens: 4096 };
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

// ─── Pipeline Passes ──────────────────────────────────────────────────────────

function validateChunkOutput(output: string, fallback: string): string {
  return output.trim().length > 0 ? output.trim() : fallback;
}

function getFastSinglePassPrompt(text: string, contentType: ContentType): string {
  const isFormal = contentType === "essay" || contentType === "academic";
  const registerNote = isFormal
    ? "REGISTER: Formal academic. No contractions. Plain academic vocabulary only — no ornate synonyms."
    : "REGISTER: Natural educated voice. Contractions welcome. Plain everyday vocabulary.";

  return `Rewrite this text to defeat AI detectors (ZeroGPT, QuillBot, GPTZero).

${registerNote}

WHAT AI DETECTORS CATCH — fix ALL of these:
1. PREDICTABLE WORD CHOICES — every word flows too smoothly from the last. Use unexpected but accurate word choices. Avoid ornate vocabulary (utilize→use, leverage→use, moreover→also, furthermore→also).
2. UNIFORM SENTENCE LENGTHS — all sentences near the same word count. Force extremes: one sentence under 8 words, one over 28 words per paragraph.
3. SIGNPOST PHRASES — remove: "furthermore", "moreover", "in conclusion", "it is important to note", "it is worth mentioning", "in today's world".
4. ABSTRACT CLAIMS — replace at least one abstract claim with a concrete specific detail.
5. REPEATED SENTENCE OPENERS — no two consecutive sentences start with the same word.
6. MISSING HUMAN VOICE — add one natural aside: "honestly", "in practice", "at least in theory", "as a rule", or "worth noting".

HARD RULES:
- Preserve ALL facts and meaning exactly
- Output ONLY the rewritten text — no preamble
- Keep result near original length (±10%)

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

function structuralPerplexityInjector(text: string, register: SourceRegister): string {
  const isFormal = register === "academic" || register === "formal";
  const parens = isFormal ? FORMAL_PARENS : CASUAL_PARENS;
  const paragraphs = text.split(/\n\s*\n/);
  let totalMods = 0;

  const result = paragraphs.map((para, pIdx) => {
    if (totalMods >= 4) return para;
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;
    let paraModCount = 0;

    const processed = sentences.map((sentence, sIdx) => {
      if (paraModCount >= 1) return sentence;
      const words = sentence.split(/\s+/).length;

      // A: Parenthetical insertion ONLY at a clause-boundary comma (followed by
      // a subordinator/conjunction). This prevents landing inside a compound
      // noun or list item.
      if (words >= 18 && sIdx >= 1 && sIdx <= 2) {
        const boundaries = findClauseBoundaryCommas(sentence);
        // Prefer a boundary roughly in the middle third of the sentence
        const candidate = boundaries.find(b => b > 15 && b < sentence.length - 25);
        if (candidate !== undefined) {
          const paren = parens[(pIdx + sIdx + totalMods) % parens.length];
          paraModCount++; totalMods++;
          return `${sentence.slice(0, candidate + 1)} ${paren}${sentence.slice(candidate + 1)}`;
        }
      }

      // B: Break "X, Y, and Z" list pattern (unchanged — operates on full
      // 3-item lists only, grammar-safe).
      if (LIST_RE.test(sentence) && paraModCount < 1) {
        const modified = sentence.replace(LIST_RE, (_, a, b, c) => `${a} and ${b} — along with ${c}`);
        if (modified !== sentence) { paraModCount++; totalMods++; return modified; }
      }

      // C: Comma → semicolon ONLY at a clause boundary comma.
      if (words >= 20 && paraModCount < 1) {
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

function interParagraphDivergencePass(text: string): string {
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

    // Strategy B: Bridge injection — only ONCE per full pass (not per
    // paragraph) to avoid literal duplicate sentences like "The data
    // confirms this." appearing multiple times.
    // Also: skip if ANY bridge phrase already exists in this paragraph
    // (prevents "That part matters. That part matters." from accumulating
    // across multiple applyDeterministicPasses calls over iterations).
    if (!didModify && modified.length >= 3 && totalInjections === 0) {
      const bridge = bridges[pIdx % bridges.length];
      const paraText = modified.join(" ").toLowerCase();
      const bridgeAlreadyPresent = bridges.some(b => paraText.includes(b.toLowerCase().slice(0, -1)));
      if (!bridgeAlreadyPresent) {
        modified.splice(1, 0, bridge);
        totalInjections++;
      }
    }

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

export function openerDiversityPass(text: string): string {
  return text.split(/\n\s*\n/).map((para) => {
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;
    let swaps = 0;
    const seen = new Map<string, number>();
    const rewritten = sentences.map((s, i) => {
      if (swaps >= 2) return s;
      const firstWord = s.split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, "");
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
  const cleaned = antiPatternPass(text, register);
  const suppressed = rhetoricalSuppressionPass(cleaned);
  const ngramBroken = zeroGPTNgramBreaker(suppressed);
  const structPerplexed = structuralPerplexityInjector(ngramBroken, register);
  const perplexed = perplexityInjector(structPerplexed, register);
  const diverged = interParagraphDivergencePass(perplexed);
  const bursty = burstinessInjector(diverged, register);
  const emdashed = emDashInjector(bursty, register);
  const opened = openerDiversityPass(emdashed);
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

// ─── Public API ───────────────────────────────────────────────────────────────

export async function humanize(
  inputText: string,
  contentType: ContentType,
  wordLimit: number
): Promise<string> {
  const truncated = truncateToWordLimit(inputText, wordLimit);
  const inputWordCount = truncated.split(/\s+/).length;

  const register = classifyRegister(truncated);

  // Short texts: single-pass humanize (fast, fits in budget)
  if (inputWordCount <= SHORT_FAST_PATH_WORDS) {
    const singlePass = await singlePassHumanize(truncated, contentType);
    return finishHumanizedText(singlePass, register, inputWordCount);
  }

  // All other texts: full 3-pass pipeline.
  // structural + semantic run in parallel per-chunk, so even 400-word texts
  // finish in ~25-35s — well within the 60s maxDuration. The mid-path
  // single-pass shortcut was removed because it sacrificed too much quality
  // (weak prompt, missed mutation pass) to save ~10s that Vercel can afford.
  const chunks = splitIntoVariableChunks(truncated);

  // Pass 1: Structural rewrite — parallel per chunk
  const structural = await structuralPass(chunks, contentType);

  // Pass 2: Semantic naturalness — parallel per chunk (all content types)
  const semantic = await semanticPass(structural, contentType);

  // Merge chunks
  const merged = semantic.join("\n\n");

  // Pass 3: Mutation — always run, never gate on internal score.
  // Our detectAI heuristic doesn't match ZeroGPT/QuillBot well enough to
  // reliably skip mutation. One extra Gemini call (~8s) is always worth it.
  const mutated = await mutationPass(merged, contentType);
  return finishHumanizedText(mutated, register, inputWordCount);
}
