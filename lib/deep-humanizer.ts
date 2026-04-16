// ─── Multi-Detector Deep Humanize Loop ───────────────────────────────────────
// Runs base humanize(), then iteratively scores against ALL 4 detectors and
// re-mutates targeting the worst scorer. Only runs on Fly.io (needs Playwright).

import { GoogleGenerativeAI } from "@google/generative-ai";
import { humanize, classifyRegister, type SourceRegister } from "@/lib/humanizer";
import { scoreAllDetectors, type AllScores } from "@/lib/scrapers";
import { type ContentType } from "@/prompts/pipeline";

export interface DeepHumanizeResult {
  text: string;
  finalScores: AllScores;
  iterations: number;
  scoreHistory: AllScores[];
}

// ─── Gemini Pro caller (stronger model for re-mutation) ───────────────────────

async function callGeminiPro(prompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-pro",
    generationConfig: { temperature: 0.92, topP: 0.95, maxOutputTokens: 8192 },
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

function worstDetector(scores: AllScores): string {
  const valid = getValidScores(scores);
  if (valid.length === 0) return "GPTZero";
  return valid.reduce((a, b) => (a.score >= b.score ? a : b)).name;
}

// ─── Targeted re-mutation prompts ────────────────────────────────────────────
// Each detector has different detection signals. The LLM (not hardcoded rules)
// generates novel fixes each iteration — no repeatable patterns.

function buildRemutationPrompt(
  text: string,
  scores: AllScores,
  worst: string,
  register: SourceRegister
): string {
  const scoreLines = getValidScores(scores)
    .map(s => `  - ${s.name}: ${s.score}% AI`)
    .join("\n");

  const detectorGuidance: Record<string, string> = {
    GPTZero: `GPTZero uses a language model to measure per-token perplexity and burstiness.
FIX STRATEGY:
- Force extreme sentence length variation: at least one sentence under 8 words AND one over 30 words per paragraph
- Use unexpected but contextually valid word choices (words GPT-2 wouldn't predict next)
- Add em-dash asides, parentheticals, and mid-thought qualifications
- Break smooth logical flow in one place — add a brief aside, a pivot, or a hedging remark
- Occasional short punchy sentences immediately after long complex ones`,

    ZeroGPT: `ZeroGPT is n-gram and phrase pattern based.
FIX STRATEGY:
- Eliminate any remaining AI signpost phrases: "furthermore", "moreover", "in conclusion", "it is important to note"
- Vary sentence openers — no two consecutive sentences should start similarly
- Replace passive constructions ("it has been shown") with active ones
- Break predictable subject-verb-object patterns in at least 3 sentences
- Use more concrete, specific language instead of abstract generalities`,

    QuillBot: `QuillBot AI detector is calibrated to phrase-level AI patterns.
FIX STRATEGY:
- Rewrite any sentences that follow a formulaic "claim + support + conclusion" structure
- Introduce natural variation in clause types: mix relative clauses, participial phrases, appositives
- Replace smooth transitions entirely with more abrupt or conversational ones
- Vary paragraph opening styles significantly — the first sentence of each paragraph should feel structurally different
- Remove any traces of "academic summary voice" — make observations feel firsthand`,

    Originality: `Originality.ai uses a deep learning model trained on LLM outputs.
FIX STRATEGY:
- Significantly restructure at least 2 paragraphs so their internal logic flow differs from the original
- Introduce unusual but valid lexical choices — words that are contextually correct but uncommon in AI output
- Add genuine qualifications and hedges that sound like human uncertainty ("though this isn't universal", "at least in many cases")
- Mix informal register momentarily in a formal text (a brief aside that drops formality for one clause)
- Ensure paragraphs don't all follow the same argumentative rhythm`,
  };

  const isFormal = register === "academic" || register === "formal";
  const registerNote = isFormal
    ? `REGISTER: Formal/academic. Do NOT add contractions or casual language. Preserve academic vocabulary.`
    : `REGISTER: Natural. Contractions and conversational tone are fine.`;

  return `You are rewriting text to reduce AI detection. Current scores:
${scoreLines}

WORST DETECTOR: ${worst} (${scores[worst.toLowerCase() as keyof AllScores]?.score ?? "?"}% AI)

${detectorGuidance[worst] ?? detectorGuidance.GPTZero}

${registerNote}

ABSOLUTE RULES:
- Preserve ALL factual content and meaning exactly
- Do NOT add asterisks, headers, bullet points, or formatting
- Do NOT add new information or change any facts
- Output ONLY the rewritten text, no commentary
- Maintain approximately the same word count (±10%)

TEXT:
${text}`;
}

// ─── Light deterministic cleanup after re-mutation ───────────────────────────

function lightCleanup(text: string): string {
  return text
    // Remove residual AI connectors at sentence starts
    .replace(/^(Furthermore|Moreover|Additionally|Consequently|In conclusion|It is important to note that|It is worth noting that),?\s*/gim, "")
    // Normalize multiple blank lines
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// ─── Main deep humanize function ─────────────────────────────────────────────

export async function humanizeDeep(
  inputText: string,
  contentType: ContentType,
  wordLimit: number,
  options: { threshold?: number; maxIterations?: number } = {}
): Promise<DeepHumanizeResult> {
  const { threshold = 15, maxIterations = 4 } = options;
  const register = classifyRegister(inputText);

  // Step 1: Base humanize
  let current = await humanize(inputText, contentType, wordLimit);

  const scoreHistory: AllScores[] = [];
  let best: { text: string; maxScore: number } = { text: current, maxScore: 100 };

  for (let iter = 0; iter < maxIterations; iter++) {
    // Step 2: Score all 4 detectors in parallel (shared browser)
    const scores = await scoreAllDetectors(current);
    scoreHistory.push(scores);

    const max = maxScore(scores);

    // Track best version seen
    if (max !== -1 && max < best.maxScore) {
      best = { text: current, maxScore: max };
    } else if (iter === 0) {
      best = { text: current, maxScore: max === -1 ? 100 : max };
    }

    // Done if all detectors below threshold
    if (max !== -1 && max <= threshold) break;

    // Last iteration — don't re-mutate
    if (iter === maxIterations - 1) break;

    // Step 3: Build targeted re-mutation prompt
    const worst = worstDetector(scores);
    const prompt = buildRemutationPrompt(current, scores, worst, register);

    // Step 4: Strong model re-mutation
    const remutated = await callGeminiPro(prompt);
    if (!remutated || remutated.trim().length < 50) continue;

    // Step 5: Light cleanup
    current = lightCleanup(remutated);
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
