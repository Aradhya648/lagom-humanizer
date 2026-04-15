// ─── GPTZero Feedback Loop ────────────────────────────────────────────────────
// Kept in a separate file so Playwright never enters the Edge runtime bundle
// used by /api/humanize. Only /api/humanize-loop (serverless) imports this.

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
import { scrapeGPTZero } from "@/lib/gptzero-scraper";
import { type ContentType } from "@/prompts/pipeline";

export interface HumanizeLoopResult {
  text: string;
  gptzeroScore: number;   // 0–100 (% AI detected), -1 if scraper failed
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

  const register = classifyRegister(inputText);
  const scoreHistory: number[] = [];
  let best = { text: "", score: 100 };
  let current = await humanize(inputText, contentType, wordLimit);

  for (let iter = 0; iter < maxIterations; iter++) {
    const result = await scrapeGPTZero(current);
    const score = result.score;
    scoreHistory.push(score);

    if (score !== -1 && score < best.score) {
      best = { text: current, score };
    } else if (iter === 0) {
      best = { text: current, score: score === -1 ? 100 : score };
    }

    if (score !== -1 && score <= threshold) break;
    if (score === -1) break;
    if (iter === maxIterations - 1) break;

    // Re-mutate targeting GPTZero signals
    const sentences = current.split(/(?<=[.!?])\s+(?=[A-Z"'])/);
    const avgLen = sentences.reduce((a, s) => a + s.split(/\s+/).length, 0) / Math.max(sentences.length, 1);

    const { GoogleGenerativeAI } = await import("@google/generative-ai");
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash",
      generationConfig: { temperature: 0.92, topP: 0.95, maxOutputTokens: 4096 },
    });

    const reMutationPrompt = `You are fixing AI-detection signals in text scoring ${score}% AI on GPTZero.

FIX:
- Sentence lengths too uniform (avg: ${Math.round(avgLen)} words) — vary aggressively
- Consecutive sentences starting with same word — change openers
- Connector words at sentence starts (However, Moreover, Furthermore) — remove or rephrase
- Overly smooth transitions — add a parenthetical or abrupt pivot

RULES:
- Preserve all factual content and meaning exactly
- Do NOT make casual/informal if text is ${register} register
- No asterisks, headers, or formatting
- Return ONLY the rewritten text

TEXT:
${current}`;

    const response = await model.generateContent(reMutationPrompt);
    const remutated = response.response.text().trim() || current;

    // Re-apply key deterministic passes
    current = openerDiversityPass(
      burstinessInjector(
        perplexityInjector(
          zeroGPTNgramBreaker(
            rhetoricalSuppressionPass(
              antiPatternPass(remutated, register)
            )
          ),
          register
        ),
        register
      )
    );
  }

  return {
    text: best.text || current,
    gptzeroScore: best.score,
    iterations: scoreHistory.length,
    scoreHistory,
  };
}
