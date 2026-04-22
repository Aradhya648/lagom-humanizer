// ─── Multi-Detector Deep Humanize Loop ───────────────────────────────────────
// Runs base humanize(), then iteratively scores against ALL 4 detectors.
// Uses TARGETED SURGERY: extracts flagged sentences from each detector,
// rewrites only those sentences, splices back. Falls back to full-text
// re-mutation if no flagged sentences available.
// Emits live events via onEvent callback (for SSE streaming to client).

import { GoogleGenerativeAI } from "@google/generative-ai";
import {
  humanize,
  classifyRegister,
  antiPatternPass,
  rhetoricalSuppressionPass,
  zeroGPTNgramBreaker,
  perplexityInjector,
  burstinessInjector,
  openerDiversityPass,
  type SourceRegister,
} from "@/lib/humanizer";
import { scoreAllDetectors, type AllScores } from "@/lib/scrapers";
import { type ContentType } from "@/prompts/pipeline";

export interface DeepHumanizeResult {
  text: string;
  finalScores: AllScores;
  iterations: number;
  scoreHistory: AllScores[];
}

export type DeepEvent =
  | { type: "status"; message: string }
  | { type: "score"; iteration: number; scores: AllScores; bestCombined: number }
  | { type: "result"; text: string; finalScores: AllScores; iterations: number; scoreHistory: AllScores[] }
  | { type: "error"; message: string };

export type EventEmitter = (event: DeepEvent) => void;

const GEMINI_TIMEOUT_MS = 30_000;
// Railway has limited RAM — 2 concurrent Playwright sessions (8 Chromium contexts)
// was causing OOM / hangs. Serialize for stability; queue the rest.
const MAX_DEEP_SESSIONS = 1;
// Absolute ceiling on a single deep session. If anything hangs past this,
// release the slot so later requests aren't blocked forever.
const DEEP_SESSION_HARD_TIMEOUT_MS = 6 * 60_000; // 6 minutes
let activeDeepSessions = 0;
const deepSessionWaiters: Array<() => void> = [];

// ─── Banned vocab (mirrors prompts/pipeline.ts) ──────────────────────────────

const BANNED_VOCAB_PROMPT = `
CRITICAL PLAIN-LANGUAGE RULE:
NEVER use these AI-paraphrased words: interrogate, employ, utilize, leverage,
furnish, bolster, ameliorate, engender, elucidate, attenuate, curtail,
culminate, underscore, foreground, substantiate, corroborate, probe, unpack,
pervasive, colossal, labyrinthine, unceasing, meteoric, insidious, unyielding,
synergistic, paradigm, multifaceted, quintessential, paramount, discernible,
startling, sophisticated, profound, pivotal, consequential, weighty, nuanced,
transformative, groundbreaking, tapestry of, constellation of, plethora of,
myriad, quandary, "navigate the complexities of".

USE INSTEAD: use, give, show, cut, many, clear, key, complex, fast, wide,
subtle, approach, question, problem, path. The shortest common word that fits
is always correct. Preserve technical domain terms (medical, scientific, legal).
`;

// ─── Gemini callers ──────────────────────────────────────────────────────────

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

async function acquireDeepSession(onEvent: EventEmitter): Promise<() => void> {
  if (activeDeepSessions >= MAX_DEEP_SESSIONS) {
    onEvent({ type: "status", message: "Queue: waiting for a free slot..." });
    await new Promise<void>((resolve) => {
      deepSessionWaiters.push(resolve);
    });
  }

  activeDeepSessions++;

  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    activeDeepSessions = Math.max(0, activeDeepSessions - 1);
    const next = deepSessionWaiters.shift();
    if (next) next();
  };

  // Safety net: auto-release the slot after the hard timeout, even if the
  // owning request is still running. Prevents permanently stuck queue.
  const autoRelease = setTimeout(() => {
    if (!released) {
      console.log(`[deep session] auto-release after ${DEEP_SESSION_HARD_TIMEOUT_MS}ms — slot was held too long`);
      release();
    }
  }, DEEP_SESSION_HARD_TIMEOUT_MS);
  // Don't keep the event loop alive just for this.
  if (typeof autoRelease.unref === "function") autoRelease.unref();

  return () => {
    clearTimeout(autoRelease);
    release();
  };
}

async function callGeminiPro(prompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-flash",
    generationConfig: { temperature: 0.88, topP: 0.95, maxOutputTokens: 8192 },
  });

  const result = await withTimeout(
    model.generateContent(prompt),
    GEMINI_TIMEOUT_MS,
    "Gemini pro call"
  );
  return result.response.text().trim();
}

async function callGeminiFlash(prompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-flash",
    generationConfig: { temperature: 0.92, topP: 0.95, maxOutputTokens: 4096 },
  });

  const result = await withTimeout(
    model.generateContent(prompt),
    GEMINI_TIMEOUT_MS,
    "Gemini flash call"
  );
  return result.response.text().trim();
}

// ─── Score utilities ─────────────────────────────────────────────────────────

function getValidScores(scores: AllScores): { name: string; score: number }[] {
  return [
    { name: "GPTZero", score: scores.gptzero.score },
    { name: "ZeroGPT", score: scores.zerogpt.score },
    { name: "QuillBot", score: scores.quillbot.score },
    { name: "Originality", score: scores.originality.score },
  ].filter(s => s.score !== -1);
}

function maxScore(scores: AllScores): number {
  const valid = getValidScores(scores);
  return valid.length > 0 ? Math.max(...valid.map(s => s.score)) : -1;
}

function combinedScore(scores: AllScores): number {
  const valid = getValidScores(scores);
  return valid.length > 0 ? valid.reduce((sum, s) => sum + s.score, 0) / valid.length : 100;
}

function worstDetector(scores: AllScores): string {
  const valid = getValidScores(scores);
  if (valid.length === 0) return "GPTZero";
  return valid.reduce((a, b) => (a.score >= b.score ? a : b)).name;
}

// ─── Detector-specific sentence rewrite prompts ──────────────────────────────

const DETECTOR_STRATEGIES: Record<string, string> = {
  GPTZero: `GPTZero's 4.4b model flagged this as "Possible AI Paraphrasing".
FIX: Scrub ornate vocabulary. Add an em-dash aside or parenthetical.
Vary sentence length — make it noticeably shorter OR longer than neighbors.
Change the opening word.`,

  ZeroGPT: `ZeroGPT flagged this sentence as AI based on phrase patterns.
FIX: Remove any signpost phrases (furthermore, moreover, in conclusion, it is important).
Rewrite with a distinct syntactic structure.
Use active voice. Change the opening word.`,

  QuillBot: `QuillBot flagged this as AI-refined text.
FIX: Break the claim-support-conclusion pattern.
Use a different clause structure (relative clause, participial phrase, or fragment).
Inject a specific detail or observation.`,

  Originality: `Originality.ai's deep model flagged this as AI.
FIX: Restructure completely. Change voice (passive↔active).
Replace abstract claims with concrete specifics.
Add a human voice marker (honestly, in practice, as a rule).`,
};

// ─── Fuzzy sentence matching + splicing ──────────────────────────────────────

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").replace(/[^\w\s]/g, "").trim();
}

function findSentenceInText(text: string, target: string): { start: number; end: number } | null {
  // 1. Exact match
  const exactIdx = text.indexOf(target);
  if (exactIdx !== -1) return { start: exactIdx, end: exactIdx + target.length };

  // 2. Normalized comparison — walk through text sentences
  const targetNorm = normalize(target);
  if (targetNorm.length < 20) return null;

  // Split text into sentence-like chunks
  const sentencePattern = /[^.!?]+[.!?]+/g;
  let match;
  while ((match = sentencePattern.exec(text)) !== null) {
    const chunk = match[0];
    const chunkNorm = normalize(chunk);
    if (chunkNorm === targetNorm) {
      return { start: match.index, end: match.index + chunk.length };
    }
    // Fuzzy: target is contained in chunk or vice versa (at least 80% overlap)
    if (chunkNorm.length > targetNorm.length * 0.8 &&
        chunkNorm.length < targetNorm.length * 1.3) {
      const overlap = longestCommonSubstring(chunkNorm, targetNorm);
      if (overlap >= Math.min(chunkNorm.length, targetNorm.length) * 0.75) {
        return { start: match.index, end: match.index + chunk.length };
      }
    }
  }

  return null;
}

function longestCommonSubstring(a: string, b: string): number {
  if (!a.length || !b.length) return 0;
  const dp: number[][] = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  let max = 0;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      if (a[i-1] === b[j-1]) {
        dp[i][j] = dp[i-1][j-1] + 1;
        if (dp[i][j] > max) max = dp[i][j];
      }
    }
  }
  return max;
}

function spliceSentence(text: string, target: string, replacement: string): string | null {
  const loc = findSentenceInText(text, target);
  if (!loc) return null;
  return text.slice(0, loc.start) + replacement + text.slice(loc.end);
}

// ─── Flagged sentence rewrite ────────────────────────────────────────────────

async function rewriteFlaggedSentence(
  sentence: string,
  surroundingContext: string,
  detectorName: string,
  register: SourceRegister,
): Promise<string> {
  const strategy = DETECTOR_STRATEGIES[detectorName] ?? DETECTOR_STRATEGIES.GPTZero;

  const isFormal = register === "academic" || register === "formal";
  const registerNote = isFormal
    ? "REGISTER: Formal/academic. No contractions. No slang. Plain academic vocabulary only."
    : "REGISTER: Natural educated voice. Contractions welcome. Plain everyday vocabulary.";

  const prompt = `You are rewriting a single sentence flagged as AI by ${detectorName}.

${strategy}

${registerNote}

${BANNED_VOCAB_PROMPT}

CONTEXT (surrounding paragraph — for meaning preservation, NOT for rewriting):
"""
${surroundingContext}
"""

SENTENCE TO REWRITE:
"""
${sentence}
"""

REQUIREMENTS:
- Preserve the exact meaning — no added or removed facts
- Output ONE rewritten sentence ONLY
- No preamble, no quotes, no commentary
- Should flow naturally where the original sat

REWRITTEN SENTENCE:`;

  const result = await callGeminiFlash(prompt);
  // Clean up: remove quotes, preambles
  return result
    .replace(/^["'`]|["'`]$/g, "")
    .replace(/^(Here(?:'s| is).*?:|Rewritten sentence:|Sentence:)\s*/i, "")
    .trim();
}

// ─── Full-text fallback re-mutation ──────────────────────────────────────────

function buildFullTextRemutationPrompt(
  text: string,
  scores: AllScores,
  worst: string,
  register: SourceRegister,
  targetWordCount: number,
): string {
  const scoreLines = getValidScores(scores)
    .map(s => `  - ${s.name}: ${s.score}% AI`)
    .join("\n");

  const currentWordCount = text.trim().split(/\s+/).length;
  const strategy = DETECTOR_STRATEGIES[worst] ?? DETECTOR_STRATEGIES.GPTZero;

  const isFormal = register === "academic" || register === "formal";
  const registerNote = isFormal
    ? "REGISTER: Formal/academic. No contractions. No slang."
    : "REGISTER: Natural educated voice. Contractions welcome.";

  return `You are rewriting text to defeat AI detectors. Current scores:
${scoreLines}

WORST DETECTOR: ${worst} — fix this first.

${strategy}

${BANNED_VOCAB_PROMPT}

${registerNote}

STRUCTURAL CHANGES — do ALL of these:
- Every paragraph MUST contain one sentence under 9 words AND one over 28 words
- Gap between shortest and longest sentence ≥ 18 words per paragraph
- No two consecutive sentences can start with the same word
- Break at least one compound sentence into two; merge at least two short ones
- Add one parenthetical aside (commas or em-dashes) per paragraph
- Replace any abstract claim with a concrete specific where natural

WORD COUNT — CRITICAL:
Current text: ${currentWordCount} words.
Target: ${targetWordCount} words (±5%).

ABSOLUTE RULES:
- Preserve ALL factual content exactly
- No asterisks, headers, bullets, or markdown
- Output ONLY the rewritten text — no preamble

TEXT:
${text}`;
}

// ─── Post-mutation cleanup passes ────────────────────────────────────────────

function applyDeterministicPasses(text: string, register: SourceRegister): string {
  let result = text
    .replace(/^(Furthermore|Moreover|Additionally|Consequently|In conclusion|It is important to note that|It is worth noting that|Importantly|Interestingly),?\s*/gim, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  result = antiPatternPass(result, register);
  result = rhetoricalSuppressionPass(result);
  result = zeroGPTNgramBreaker(result);
  result = perplexityInjector(result, register);
  result = burstinessInjector(result, register);
  result = openerDiversityPass(result);
  return result;
}

// ─── Helpers to extract & dedup flagged sentences ────────────────────────────

function collectFlaggedSentences(scores: AllScores): Array<{ sentence: string; detectors: string[] }> {
  const map = new Map<string, Set<string>>();  // normalized sentence → set of detectors

  const addFrom = (detector: string, sentences: string[]) => {
    for (const s of sentences) {
      const norm = normalize(s);
      if (norm.length < 20) continue;
      if (!map.has(norm)) map.set(norm, new Set());
      map.get(norm)!.add(detector);
      // Store original form as the key marker
      map.get(norm)!.add(`__orig:${s}`);
    }
  };

  addFrom("GPTZero", scores.gptzero.flaggedSentences);
  addFrom("ZeroGPT", scores.zerogpt.flaggedSentences);
  addFrom("QuillBot", scores.quillbot.flaggedSentences);
  addFrom("Originality", scores.originality.flaggedSentences);

  const results: Array<{ sentence: string; detectors: string[] }> = [];
  for (const [, tags] of map) {
    const origMarker = Array.from(tags).find(t => t.startsWith("__orig:"));
    const detectors = Array.from(tags).filter(t => !t.startsWith("__orig:"));
    if (origMarker && detectors.length > 0) {
      results.push({ sentence: origMarker.slice(7), detectors });
    }
  }

  // Prioritize sentences flagged by multiple detectors
  results.sort((a, b) => b.detectors.length - a.detectors.length);
  return results;
}

function getSurroundingContext(text: string, sentence: string): string {
  const loc = findSentenceInText(text, sentence);
  if (!loc) return sentence;
  // Grab ~200 chars before and after
  const start = Math.max(0, loc.start - 200);
  const end = Math.min(text.length, loc.end + 200);
  return text.slice(start, end);
}

// ─── Main deep humanize function ─────────────────────────────────────────────

export async function humanizeDeep(
  inputText: string,
  contentType: ContentType,
  wordLimit: number,
  options: { threshold?: number; maxIterations?: number; abortSignal?: AbortSignal } = {},
  onEvent: EventEmitter = () => {}
): Promise<DeepHumanizeResult> {
  const releaseSession = await acquireDeepSession(onEvent);

  try {
    // maxIterations: 3 is the sweet spot. Empirically, most score improvement
    // lands in rounds 1-2; rounds 4+ see diminishing returns AND blow past
    // the route's 5-min hard timeout on longer inputs (800+ words → each
    // round is ~90-120s between Playwright scoring + Gemini rewrites).
    const { threshold = 5, maxIterations = 3, abortSignal } = options;
    const register = classifyRegister(inputText);
    const originalWordCount = inputText.trim().split(/\s+/).length;
    const targetWordCount = Math.min(originalWordCount, wordLimit);

    onEvent({ type: "status", message: "Running base humanization pipeline..." });
    let current = await humanize(inputText, contentType, wordLimit);

    const scoreHistory: AllScores[] = [];
    let best: { text: string; combined: number; scores: AllScores | null } = {
      text: current,
      combined: 100,
      scores: null,
    };

    for (let iter = 0; iter < maxIterations; iter++) {
      // Soft-abort: route handler signals us ~30s before the hard timeout
      // so we can stop cleanly and emit the best-so-far result.
      if (abortSignal?.aborted) {
        onEvent({ type: "status", message: `Stopping early to return best result so far...` });
        break;
      }

      onEvent({
        type: "status",
        message: `Round ${iter + 1}/${maxIterations}: scoring against all 4 detectors...`,
      });

      const scores = await scoreAllDetectors(current);
      scoreHistory.push(scores);

      const max = maxScore(scores);
      const avg = combinedScore(scores);

      if (avg < best.combined) {
        best = { text: current, combined: avg, scores };
      }

      onEvent({ type: "score", iteration: iter + 1, scores, bestCombined: best.combined });

      console.log(`[deep iter ${iter + 1}] max=${max} avg=${avg.toFixed(0)} words=${current.split(/\s+/).length}`);

      // Done if we got ANY valid score AND all valid scores are ≤ threshold.
      // (Previously required a "heavyweight" which caused infinite loops when
      // GPTZero/Originality were flaky — any working detector is now enough.)
      const validScores = getValidScores(scores);
      if (validScores.length > 0 && max !== -1 && max <= threshold) {
        onEvent({ type: "status", message: `All working detectors ≤ ${threshold}% — done.` });
        break;
      }
      if (validScores.length === 0) {
        onEvent({ type: "status", message: `No detectors returned valid scores this round — continuing with internal heuristics...` });
      }

      // Last iteration — no more work
      if (iter === maxIterations - 1) break;

      // ── TARGETED SURGERY: rewrite only flagged sentences ────────────────────
      // Dedup normalized sentences AND cap the total to avoid blow-ups.
      const flaggedRaw = collectFlaggedSentences(scores);
      const seenNorms = new Set<string>();
      const flagged: typeof flaggedRaw = [];
      for (const f of flaggedRaw) {
        const n = normalize(f.sentence);
        if (n.length < 20 || seenNorms.has(n)) continue;
        seenNorms.add(n);
        flagged.push(f);
        if (flagged.length >= 5) break; // cap parallel rewrites
      }
      const worst = worstDetector(scores);

      if (flagged.length > 0) {
        onEvent({
          type: "status",
          message: `Found ${flagged.length} flagged sentence${flagged.length === 1 ? "" : "s"} — rewriting in parallel...`,
        });

        // Rewrite ALL flagged sentences in parallel with allSettled so a
        // single hung Gemini call can't stall the round. withTimeout inside
        // callGeminiFlash caps each call at GEMINI_TIMEOUT_MS.
        const rewrites = await Promise.allSettled(
          flagged.map(({ sentence, detectors }) => {
            const targetDetector = detectors[0] ?? worst;
            const context = getSurroundingContext(current, sentence);
            return rewriteFlaggedSentence(sentence, context, targetDetector, register)
              .then(newSentence => ({ sentence, newSentence }));
          }),
        );

        // Splice one at a time, guarding against duplicate insertions.
        let rewritten = current;
        let spliceCount = 0;
        const splicedOriginals = new Set<string>();
        const splicedReplacements = new Set<string>();

        for (const r of rewrites) {
          if (r.status !== "fulfilled") {
            console.log(`[deep surgery] rewrite rejected: ${r.reason?.message ?? r.reason}`);
            continue;
          }
          const { sentence, newSentence } = r.value;
          if (!newSentence || newSentence.length < 10) continue;

          const origNorm = normalize(sentence);
          const newNorm = normalize(newSentence);
          // Skip if we already spliced this original, or if the "new"
          // sentence is something we've already inserted (prevents the
          // "This distinction matters. This distinction matters." bug).
          if (splicedOriginals.has(origNorm)) continue;
          if (splicedReplacements.has(newNorm)) continue;
          if (origNorm === newNorm) continue; // no-op rewrite

          const spliced = spliceSentence(rewritten, sentence, newSentence);
          if (spliced && spliced !== rewritten) {
            rewritten = spliced;
            spliceCount++;
            splicedOriginals.add(origNorm);
            splicedReplacements.add(newNorm);
          }
        }

        onEvent({
          type: "status",
          message: `Spliced ${spliceCount} rewritten sentences. Cleaning up...`,
        });

        current = applyDeterministicPasses(rewritten, register);
      } else {
        // ── FALLBACK: full-text re-mutation ────────────────────────────────────
        onEvent({
          type: "status",
          message: `No flagged sentences extractable — targeting ${worst} (${scores[worstDetectorKey(worst)]?.score ?? 0}%) with full rewrite...`,
        });

        const currentWordCount = current.trim().split(/\s+/).length;
        const aim = Math.max(targetWordCount, Math.round(currentWordCount * 0.97));
        const prompt = buildFullTextRemutationPrompt(current, scores, worst, register, aim);

        try {
          const remutated = await callGeminiPro(prompt);
          if (remutated && remutated.trim().length > 50) {
            current = applyDeterministicPasses(remutated.trim(), register);
          }
        } catch (err) {
          console.log(`[deep full-mutation] failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }

    const finalScores = best.scores ?? scoreHistory[scoreHistory.length - 1] ?? {
      gptzero: { score: -1, label: "skipped", flaggedSentences: [] },
      zerogpt: { score: -1, label: "skipped", flaggedSentences: [] },
      quillbot: { score: -1, label: "skipped", flaggedSentences: [] },
      originality: { score: -1, label: "skipped", flaggedSentences: [] },
    };

    const result: DeepHumanizeResult = {
      text: best.text,
      finalScores,
      iterations: scoreHistory.length,
      scoreHistory,
    };

    onEvent({
      type: "result",
      text: result.text,
      finalScores: result.finalScores,
      iterations: result.iterations,
      scoreHistory: result.scoreHistory,
    });

    return result;
  } finally {
    releaseSession();
  }
}

function worstDetectorKey(name: string): keyof AllScores {
  const map: Record<string, keyof AllScores> = {
    GPTZero: "gptzero",
    ZeroGPT: "zerogpt",
    QuillBot: "quillbot",
    Originality: "originality",
  };
  return map[name] ?? "gptzero";
}
