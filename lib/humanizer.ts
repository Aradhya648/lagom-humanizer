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
}

const STRUCTURAL_SETTINGS: GenSettings = { temperature: 0.85, topP: 0.95 };
const SEMANTIC_SETTINGS: GenSettings  = { temperature: 0.75, topP: 0.90 };
const MUTATION_SETTINGS: GenSettings  = { temperature: 0.88, topP: 0.92 };

// Stronger model for the mutation pass — better instruction following
const MUTATION_MODEL = "gemini-2.5-pro";

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
      maxOutputTokens: 8192,
    },
  });
  const result = await model.generateContent(getMutationPrompt(text, contentType));
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

// ─── Perplexity Injector ─────────────────────────────────────────────────────
// ZeroGPT is perplexity-based. Swaps up to 3 common words per paragraph with
// lower-frequency synonyms from a curated pool to raise per-token surprise.
// Register-gated: formalSafe pool for academic/formal, full pool otherwise.

interface SynonymSwap { word: string; replacements: string[]; formalSafe: boolean; }

const PERPLEXITY_SWAP_POOL: SynonymSwap[] = [
  { word: "shows",     replacements: ["illustrates", "reveals", "makes clear"],           formalSafe: true  },
  { word: "show",      replacements: ["illustrate", "reveal", "make clear"],              formalSafe: true  },
  { word: "shown",     replacements: ["illustrated", "borne out", "corroborated"],        formalSafe: true  },
  { word: "uses",      replacements: ["employs", "draws on", "leverages"],                formalSafe: true  },
  { word: "use",       replacements: ["employ", "draw on", "leverage"],                   formalSafe: true  },
  { word: "highlight", replacements: ["underscore", "foreground", "bring into focus"],    formalSafe: true  },
  { word: "suggest",   replacements: ["intimate", "point toward", "hint at"],             formalSafe: true  },
  { word: "indicate",  replacements: ["signal", "attest to", "speak to"],                 formalSafe: true  },
  { word: "examine",   replacements: ["interrogate", "probe", "unpack"],                  formalSafe: true  },
  { word: "improve",   replacements: ["sharpen", "bolster", "refine"],                    formalSafe: true  },
  { word: "increase",  replacements: ["amplify", "heighten", "compound"],                 formalSafe: true  },
  { word: "reduce",    replacements: ["attenuate", "curtail", "pare down"],               formalSafe: true  },
  { word: "address",   replacements: ["contend with", "grapple with", "take up"],         formalSafe: true  },
  { word: "many",      replacements: ["numerous", "an array of", "a range of"],           formalSafe: true  },
  { word: "important", replacements: ["consequential", "pivotal", "weighty"],             formalSafe: true  },
  { word: "different", replacements: ["distinct", "divergent", "varying"],                formalSafe: true  },
  { word: "complex",   replacements: ["multifaceted", "intricate", "layered"],            formalSafe: true  },
  { word: "clear",     replacements: ["apparent", "discernible", "evident"],              formalSafe: true  },
  { word: "often",     replacements: ["frequently", "in many cases", "not infrequently"], formalSafe: true  },
  { word: "also",      replacements: ["too", "as well", "equally"],                       formalSafe: true  },
  { word: "very",      replacements: ["considerably", "markedly", "rather"],              formalSafe: true  },
  { word: "because",   replacements: ["given that", "insofar as", "since"],               formalSafe: true  },
  { word: "problem",   replacements: ["difficulty", "complication", "obstacle"],          formalSafe: true  },
  { word: "question",  replacements: ["puzzle", "quandary", "matter"],                    formalSafe: true  },
  { word: "approach",  replacements: ["tack", "avenue", "orientation"],                   formalSafe: true  },
  { word: "get",       replacements: ["obtain", "acquire", "land"],                       formalSafe: false },
  { word: "good",      replacements: ["solid", "worthwhile", "capable"],                  formalSafe: false },
  { word: "big",       replacements: ["sizeable", "outsized", "substantial"],             formalSafe: false },
  { word: "small",     replacements: ["slim", "modest", "narrow"],                        formalSafe: false },
];

function perplexityInjector(text: string, register: SourceRegister): string {
  const isFormal = register === "academic" || register === "formal";
  const pool = isFormal ? PERPLEXITY_SWAP_POOL.filter(s => s.formalSafe) : PERPLEXITY_SWAP_POOL;
  return text.split(/\n\s*\n/).map((para) => {
    let swapCount = 0;
    let current = para;
    for (const swap of pool) {
      if (swapCount >= 5) break;
      const re = new RegExp(`\\b${swap.word}\\b`, "i");
      if (!re.test(current)) continue;
      const idx = (current.length + swapCount) % swap.replacements.length;
      const replacement = swap.replacements[idx];
      current = current.replace(re, (m) =>
        m[0] === m[0].toUpperCase() && m[0] !== m[0].toLowerCase()
          ? replacement[0].toUpperCase() + replacement.slice(1) : replacement
      );
      swapCount++;
    }
    return current;
  }).join("\n\n");
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

function zeroGPTNgramBreaker(text: string): string {
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

function structuralPerplexityInjector(text: string, register: SourceRegister): string {
  const isFormal = register === "academic" || register === "formal";
  const parens = isFormal ? FORMAL_PARENS : CASUAL_PARENS;
  const paragraphs = text.split(/\n\s*\n/);
  let totalMods = 0;

  const result = paragraphs.map((para, pIdx) => {
    if (totalMods >= 5) return para;
    const sentences = getSentences(para);
    if (sentences.length < 2) return para;
    let paraModCount = 0;

    const processed = sentences.map((sentence, sIdx) => {
      if (paraModCount >= 2) return sentence;
      const words = sentence.split(/\s+/).length;

      // A: Parenthetical insertion after first comma in 18+ word sentences
      if (words >= 18 && sIdx >= 1 && sIdx <= 2 && paraModCount === 0) {
        const commaPos = sentence.indexOf(",");
        if (commaPos > 10 && commaPos < sentence.length - 20) {
          const paren = parens[(pIdx + sIdx + totalMods) % parens.length];
          paraModCount++; totalMods++;
          return `${sentence.slice(0, commaPos + 1)} ${paren}${sentence.slice(commaPos + 1)}`;
        }
      }

      // B: Break "X, Y, and Z" list pattern
      if (LIST_RE.test(sentence) && paraModCount < 2) {
        const modified = sentence.replace(LIST_RE, (_, a, b, c) => `${a} and ${b} — along with ${c}`);
        if (modified !== sentence) { paraModCount++; totalMods++; return modified; }
      }

      // C: Comma → semicolon in long sentences
      if (words >= 20 && paraModCount < 2) {
        const commas: number[] = [];
        let pos = 0;
        while ((pos = sentence.indexOf(",", pos)) !== -1) { commas.push(pos); pos++; }
        if (commas.length >= 2) {
          const mid = commas[Math.floor(commas.length / 2)];
          if (mid > 15 && mid < sentence.length - 15) {
            paraModCount++; totalMods++;
            return `${sentence.slice(0, mid)};${sentence.slice(mid + 1)}`;
          }
        }
      }

      return sentence;
    });

    return processed.join(" ");
  });

  return result.join("\n\n");
}

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

function burstinessInjector(text: string, register: SourceRegister): string {
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

    let modified = [...sentences];
    let didModify = false;

    // Strategy A: Split longest sentence at a comma
    const longestIdx = lengths.indexOf(Math.max(...lengths));
    if (lengths[longestIdx] > 18) {
      const commaPos = modified[longestIdx].indexOf(",", 15);
      if (commaPos > 0 && commaPos < modified[longestIdx].length - 15) {
        const first = modified[longestIdx].slice(0, commaPos).trimEnd() + ".";
        const second = modified[longestIdx].slice(commaPos + 1).trimStart();
        modified[longestIdx] = first;
        modified.splice(longestIdx + 1, 0, second.charAt(0).toUpperCase() + second.slice(1));
        didModify = true;
        totalInjections++;
      }
    }

    // Strategy B: Also inject a bridge sentence for extreme variance (GPTZero needs spiky length pattern)
    if (modified.length >= 3) {
      const bridge = bridges[(pIdx + totalInjections) % bridges.length];
      // Insert near start (after sentence 1) to create early contrast
      modified.splice(1, 0, bridge);
      if (!didModify) totalInjections++;
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
    if (totalInjections >= 3) return para;
    const sentences = getSentences(para);
    const modified = sentences.map((sentence, sIdx) => {
      if (totalInjections >= 3) return sentence;
      const words = sentence.split(/\s+/).length;
      if (words < 20) return sentence;
      // Only inject into every 2nd qualifying sentence to avoid over-formatting
      if ((pIdx + sIdx) % 2 !== 0) return sentence;

      // Find a natural injection point: after a content word around the mid-point
      const wordArr = sentence.split(/\s+/);
      const mid = Math.floor(wordArr.length * 0.45);
      // Look for a word boundary near mid that isn't a comma or conjunction
      let insertAt = -1;
      for (let offset = 0; offset <= 4; offset++) {
        const idx = mid + offset;
        if (idx < wordArr.length - 4 && !/^(and|or|but|the|a|an|of|in|to|with|for)$/i.test(wordArr[idx])) {
          insertAt = idx;
          break;
        }
      }
      if (insertAt === -1) return sentence;

      const aside = asides[(pIdx + sIdx + totalInjections) % asides.length];
      const before = wordArr.slice(0, insertAt + 1).join(" ");
      const after = wordArr.slice(insertAt + 1).join(" ");
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

function openerDiversityPass(text: string): string {
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

  // Pass 6: ZeroGPT n-gram breaker — 30 AI-pattern rules
  const ngramBroken = zeroGPTNgramBreaker(suppressed);

  // Pass 7: Structural perplexity — parentheticals, em-dashes, semicolons
  const structPerplexed = structuralPerplexityInjector(ngramBroken, register);

  // Pass 8: Word-level perplexity — up to 3 rare synonym swaps per paragraph
  const perplexed = perplexityInjector(structPerplexed, register);

  // Pass 9: Inter-paragraph divergence — Originality.ai cross-para vocabulary
  const diverged = interParagraphDivergencePass(perplexed);

  // Pass 10: GPTZero burstiness — force sentence-length variance
  const bursty = burstinessInjector(diverged, register);

  // Pass 11: Em-dash injection — raises per-token perplexity for GPTZero
  const emdashed = emDashInjector(bursty, register);

  // Pass 12: Opener diversity — no duplicate sentence-starting words
  const opened = openerDiversityPass(emdashed);

  // Pass 13: Length discipline (ceiling + floor)
  const lengthCapped = enforceLengthDiscipline(opened, inputWordCount);

  // Pass 14: Expand if LLM compressed too aggressively (< 88% of original)
  const finalWordCount = lengthCapped.split(/\s+/).length;
  if (finalWordCount < inputWordCount * 0.88) {
    const targetWords = Math.round(inputWordCount * 0.97);
    const expansionPrompt = `Expand this text to approximately ${targetWords} words by adding relevant elaboration, examples, or detail within existing sentences. Do NOT add new sections or change any facts. Preserve the exact register and style.

Output only the expanded text. No preamble.

TEXT:
${lengthCapped}`;
    const expanded = await callModel(expansionPrompt, { temperature: 0.70, topP: 0.90 });
    return enforceLengthDiscipline(expanded.trim() || lengthCapped, Math.round(inputWordCount * 1.05));
  }

  return lengthCapped;
}

// ─── GPTZero Feedback Loop ────────────────────────────────────────────────────
// Runs humanize(), checks live GPTZero score via Playwright scraper,
// and re-applies targeted mutation if score is above threshold.
// Max 3 iterations. Returns best result seen across all iterations.

export interface HumanizeLoopResult {
  text: string;
  gptzeroScore: number;        // final GPTZero score (0–100), -1 if scraper failed
  iterations: number;
  scoreHistory: number[];
}

export async function humanizeLoop(
  inputText: string,
  contentType: ContentType,
  wordLimit: number,
  options: { threshold?: number; maxIterations?: number } = {}
): Promise<HumanizeLoopResult> {
  const { threshold = 15, maxIterations = 3 } = options;

  // Lazy-import scraper so it only loads on Render (has Playwright),
  // not on Vercel builds where chromium is unavailable.
  const { scrapeGPTZero } = await import("@/lib/gptzero-scraper");

  const register = classifyRegister(inputText);
  const scoreHistory: number[] = [];
  let best = { text: "", score: 100 };
  let current = await humanize(inputText, contentType, wordLimit);

  for (let iter = 0; iter < maxIterations; iter++) {
    const result = await scrapeGPTZero(current);
    const score = result.score;
    scoreHistory.push(score);

    // Track best
    if (score !== -1 && score < best.score) {
      best = { text: current, score };
    } else if (iter === 0) {
      best = { text: current, score: score === -1 ? 100 : score };
    }

    // Done if below threshold or scraper failed
    if (score !== -1 && score <= threshold) break;
    if (score === -1) break;
    if (iter === maxIterations - 1) break;

    // Re-mutate focusing on worst sentences:
    // Split into sentences, target those most likely causing detection
    const sentences = current.split(/(?<=[.!?])\s+(?=[A-Z"'])/);
    const avgLen = sentences.reduce((a, s) => a + s.split(/\s+/).length, 0) / Math.max(sentences.length, 1);

    // Flag uniform-length runs (GPTZero signal) and re-mutate full text
    const reMutationPrompt = `You are fixing AI-detection signals in text. The text below is scoring ${score}% AI on GPTZero.

TARGET SIGNALS TO FIX:
- Sentences with identical or near-identical word counts (avg: ${Math.round(avgLen)} words) — vary lengths aggressively
- First words of consecutive sentences that are the same — change them
- Any connector words at sentence starts: However, Moreover, Furthermore, Additionally — remove or rephrase
- Overly smooth transitions between ideas — add a rough edge, mid-thought parenthetical, or abrupt pivot

RULES:
- Preserve all factual content and meaning exactly
- Do NOT add casual/informal tone if text is ${register} register
- Do NOT add asterisks, headers, or formatting
- Return ONLY the rewritten text, no commentary

TEXT:
${current}`;

    const remutated = await callGemini(reMutationPrompt, { temperature: 0.92, topP: 0.95 });

    // Apply deterministic passes again on remutated output
    const cleaned = antiPatternPass(remutated, register);
    const suppressed = rhetoricalSuppressionPass(cleaned);
    const ngramBroken = zeroGPTNgramBreaker(suppressed);
    const perplexed = perplexityInjector(ngramBroken, register);
    const bursty = burstinessInjector(perplexed, register);
    current = openerDiversityPass(bursty);
  }

  return {
    text: best.text || current,
    gptzeroScore: best.score,
    iterations: scoreHistory.length,
    scoreHistory,
  };
}
