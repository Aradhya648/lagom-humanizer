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

// Split on paragraph boundaries — used by Phase 2 chunk pipeline.
// In Phase 1 we run passes on the full text; this function is wired up but
// the output is merged back before processing.
export function splitIntoChunks(text: string): string[] {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
  return paragraphs.length > 0 ? paragraphs : [text.trim()];
}

// ─── Model Wrappers ─────────────────────────────────────────────────────────

async function callGemini(
  prompt: string,
  temperature: number,
  topP: number
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-flash",
    generationConfig: { temperature, topP, maxOutputTokens: 4096 },
  });

  const result = await model.generateContent(prompt);
  const text = result.response.text();
  if (!text || text.trim().length === 0) {
    throw new Error("Gemini returned empty response");
  }
  return text.trim();
}

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

// Primary: Gemini. Fallback: HuggingFace.
async function callModel(
  prompt: string,
  temperature: number,
  topP: number
): Promise<string> {
  try {
    return await callGemini(prompt, temperature, topP);
  } catch (geminiError) {
    console.warn("Gemini failed, falling back to HuggingFace:", geminiError);
    return await callHuggingFace(prompt);
  }
}

// ─── Pipeline Passes ────────────────────────────────────────────────────────

// Pass 1 — Structural rewrite.
// Higher temperature to introduce rhythm variation. Focus: sentence structure only.
// Runs for all modes.
async function structuralPass(
  text: string,
  mode: HumanizeMode
): Promise<string> {
  const temps: Record<HumanizeMode, number> = {
    light: 0.70,
    medium: 0.85,
    aggressive: 0.95,
  };
  return callModel(getStructuralPrompt(text, mode), temps[mode], 0.95);
}

// Pass 2 — Semantic naturalness.
// Lower temperature to preserve meaning while improving voice.
// Skipped for light mode — light rewrite doesn't need voice rework.
async function semanticPass(
  text: string,
  mode: HumanizeMode
): Promise<string> {
  if (mode === "light") return text;
  const temps: Record<HumanizeMode, number> = {
    light: 0.70,
    medium: 0.75,
    aggressive: 0.82,
  };
  return callModel(getSemanticPrompt(text, mode), temps[mode], 0.90);
}

// Pass 3 — Selective mutation.
// Only fires when the text still scores above the synthetic threshold.
// Skipped for light mode. Threshold is lower for aggressive (stricter).
async function mutationPass(
  text: string,
  mode: HumanizeMode
): Promise<string> {
  if (mode === "light") return text;

  const threshold = mode === "aggressive" ? 42 : 52;
  const { score } = detectAI(text);

  if (score <= threshold) return text; // already good enough — skip

  return callModel(getMutationPrompt(text), 0.90, 0.95);
}

// ─── Public API ─────────────────────────────────────────────────────────────

export async function humanize(
  inputText: string,
  mode: HumanizeMode,
  wordLimit: number
): Promise<string> {
  const truncated = truncateToWordLimit(inputText, wordLimit);

  // Pass 1: Fix structural patterns and sentence rhythm
  let result = await structuralPass(truncated, mode);

  // Pass 2: Improve naturalness and voice (medium + aggressive only)
  result = await semanticPass(result, mode);

  // Pass 3: Targeted mutation of remaining synthetic patterns (gated by score)
  result = await mutationPass(result, mode);

  return result;
}
