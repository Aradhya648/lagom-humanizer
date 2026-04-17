// ─── Multi-Detector Deep Humanize Loop ───────────────────────────────────────
// Runs base humanize(), then iteratively scores against ALL 4 detectors and
// re-mutates targeting the worst scorer. Only runs on Railway (needs Playwright).

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

// ─── Gemini Pro caller ────────────────────────────────────────────────────────

async function callGeminiPro(prompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-pro",
    generationConfig: { temperature: 0.88, topP: 0.95, maxOutputTokens: 8192 },
  });

  const result = await model.generateContent(prompt);
  return result.response.text().trim();
}

// ─── Score utilities ──────────────────────────────────────────────────────────

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

// ─── Targeted re-mutation prompts ────────────────────────────────────────────

function buildRemutationPrompt(
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

  const detectorGuidance: Record<string, string> = {
    GPTZero: `GPTZero uses a language model to measure per-token perplexity and burstiness.
FIX STRATEGY — do ALL of these:
- Every paragraph MUST have at least one sentence under 8 words AND one over 28 words
- Use unexpected but contextually valid word choices — uncommon collocations, vivid verbs
- Insert em-dash asides: "The effect—worth noting here—compounds quickly"
- Add a mid-paragraph parenthetical: "(though this varies considerably)"
- One abrupt pivot or self-correction: "Or rather, the more accurate framing is..."
- Mix short punchy sentences directly after long complex ones`,

    ZeroGPT: `ZeroGPT is n-gram and phrase pattern based.
FIX STRATEGY — do ALL of these:
- Remove ALL AI signpost phrases: "furthermore", "moreover", "in conclusion", "it is important", "it is worth noting"
- No two consecutive sentences can start with the same word or same syntactic structure
- Replace every passive construction with active voice
- Replace at least 3 abstract claims with specific, concrete versions
- Make transitions between ideas feel earned, not formulaic`,

    QuillBot: `QuillBot AI detector flags phrase-level AI patterns.
FIX STRATEGY — do ALL of these:
- Break every "claim → support → conclusion" sentence pattern
- Mix clause types aggressively: relative clauses, participial phrases, appositives, fragments
- Start at least one paragraph with something other than a topic sentence
- Replace smooth transitions with more abrupt or unexpected pivots
- One sentence should feel like an observation the writer just had, not a prepared point`,

    Originality: `Originality.ai uses a deep model trained on LLM outputs.
FIX STRATEGY — do ALL of these:
- Restructure 2+ paragraphs so their internal argument flow differs from input
- Use unusual but valid word combinations — "the data bristles with", "the gap yawns"
- Add hedges that sound like human uncertainty: "at least in most cases", "though I'd argue"
- Drop formality briefly in one clause even in academic text
- Vary argumentative rhythm — not every paragraph should follow the same pattern`,
  };

  const isFormal = register === "academic" || register === "formal";
  const registerNote = isFormal
    ? `REGISTER: Formal/academic. No contractions. No casual language. Preserve academic vocabulary.`
    : `REGISTER: Natural educated voice. Contractions and conversational phrasing are welcome.`;

  return `You are rewriting text to defeat AI detectors. Current scores:
${scoreLines}

WORST DETECTOR: ${worst} — fix this first.

${detectorGuidance[worst] ?? detectorGuidance.GPTZero}

${registerNote}

WORD COUNT — CRITICAL:
Current text: ${currentWordCount} words.
Target: ${targetWordCount} words (±5%).
If your output is shorter than ${Math.round(targetWordCount * 0.95)} words, expand sentences with relevant detail. Do NOT truncate content.

ABSOLUTE RULES:
- Preserve ALL factual content and meaning exactly
- No asterisks, headers, bullets, or markdown formatting
- No new facts or invented content
- Output ONLY the rewritten text — no preamble, no commentary

TEXT:
${text}`;
}

// ─── Full deterministic cleanup after each re-mutation ───────────────────────
// Re-applies every detector-specific pass so re-mutations don't undo prior gains

function applyDeterministicPasses(text: string, register: SourceRegister): string {
  let result = text
    // Kill residual AI connectors at paragraph starts
    .replace(/^(Furthermore|Moreover|Additionally|Consequently|In conclusion|It is important to note that|It is worth noting that|Importantly|Interestingly),?\s*/gim, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  result = antiPatternPass(result, register);
  result = rhetoricalSuppressionPass(result);
  result = zeroGPTNgramBreaker(result);          // Always — prevents ZeroGPT regression
  result = perplexityInjector(result, register);
  result = burstinessInjector(result, register);
  result = openerDiversityPass(result);
  return result;
}

// ─── Word count expansion if LLM compressed ──────────────────────────────────

async function expandIfShort(text: string, targetWords: number): Promise<string> {
  const currentWords = text.trim().split(/\s+/).length;
  if (currentWords >= Math.round(targetWords * 0.88)) return text; // Within range

  const expandPrompt = `The following text is ${currentWords} words but needs to be approximately ${targetWords} words.
Expand it by adding relevant elaboration, concrete examples, or supporting detail within the existing sentences and paragraphs.
Do NOT add new sections, new arguments, or change any facts.
Preserve the exact register and style.
Output only the expanded text.

TEXT:
${text}`;

  try {
    const expanded = await callGeminiPro(expandPrompt);
    return expanded.trim().length > currentWords * 4 ? text : (expanded.trim() || text);
  } catch {
    return text;
  }
}

// ─── Main deep humanize function ─────────────────────────────────────────────

export async function humanizeDeep(
  inputText: string,
  contentType: ContentType,
  wordLimit: number,
  options: { threshold?: number; maxIterations?: number } = {}
): Promise<DeepHumanizeResult> {
  const { threshold = 15, maxIterations = 3 } = options;
  const register = classifyRegister(inputText);
  const originalWordCount = inputText.trim().split(/\s+/).length;
  const targetWordCount = Math.min(originalWordCount, wordLimit);

  // Step 1: Base humanize (handles structural + semantic + deterministic passes)
  let current = await humanize(inputText, contentType, wordLimit);

  const scoreHistory: AllScores[] = [];
  let best: { text: string; combined: number } = { text: current, combined: 100 };

  for (let iter = 0; iter < maxIterations; iter++) {
    // Step 2: Score all 4 detectors in parallel (shared browser, 4 tabs)
    const scores = await scoreAllDetectors(current);
    scoreHistory.push(scores);

    const max = maxScore(scores);
    const avg = combinedScore(scores);

    // Track best version by combined average score
    if (avg < best.combined) {
      best = { text: current, combined: avg };
    }

    console.log(`[deep iter ${iter + 1}] max=${max} avg=${avg.toFixed(0)} words=${current.split(/\s+/).length}`);

    // Done if all valid detectors below threshold
    if (max !== -1 && max <= threshold) break;

    // Last iteration — stop here
    if (iter === maxIterations - 1) break;

    // Step 3: Build targeted re-mutation prompt for worst detector
    const worst = worstDetector(scores);
    const currentWordCount = current.trim().split(/\s+/).length;
    // Aim for the greater of original target or what we currently have
    const aim = Math.max(targetWordCount, Math.round(currentWordCount * 0.97));
    const prompt = buildRemutationPrompt(current, scores, worst, register, aim);

    // Step 4: Strong model re-mutation
    const remutated = await callGeminiPro(prompt);
    if (!remutated || remutated.trim().length < 50) continue;

    // Step 5: Expand if LLM compressed too much
    const expanded = await expandIfShort(remutated.trim(), aim);

    // Step 6: Re-apply full deterministic pipeline
    // This prevents each re-mutation from undoing ZeroGPT/QuillBot gains
    current = applyDeterministicPasses(expanded, register);
  }

  const finalScores = scoreHistory[scoreHistory.length - 1] ?? {
    gptzero: { score: -1, label: "skipped" },
    zerogpt: { score: -1, label: "skipped" },
    quillbot: { score: -1, label: "skipped" },
    originality: { score: -1, label: "skipped" },
  };

  return {
    text: best.text,
    finalScores,
    iterations: scoreHistory.length,
    scoreHistory,
  };
}
