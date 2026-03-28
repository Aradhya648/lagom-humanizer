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

// Pass 1 — Structural rewrite per chunk.
// Parallel across chunks. Each chunk gets a variation hint via chunkIndex.
// Higher temperature to introduce rhythm variation.
async function structuralPass(
  chunks: string[],
  mode: HumanizeMode
): Promise<string[]> {
  const temps: Record<HumanizeMode, number> = {
    light: 0.70,
    medium: 0.85,
    aggressive: 0.95,
  };
  const temp = temps[mode];

  return Promise.all(
    chunks.map((chunk, i) =>
      callModel(getStructuralPrompt(chunk, mode, i), temp, 0.95)
    )
  );
}

// Pass 2 — Semantic naturalness per chunk.
// Parallel across chunks. Lower temperature to preserve meaning.
// Skipped entirely for light mode.
async function semanticPass(
  chunks: string[],
  mode: HumanizeMode
): Promise<string[]> {
  if (mode === "light") return chunks;

  const temps: Record<HumanizeMode, number> = {
    light: 0.70,
    medium: 0.75,
    aggressive: 0.82,
  };
  const temp = temps[mode];

  return Promise.all(
    chunks.map((chunk, i) =>
      callModel(getSemanticPrompt(chunk, mode, i), temp, 0.90)
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

  return callModel(getMutationPrompt(text), 0.90, 0.95);
}

// ─── Public API ─────────────────────────────────────────────────────────────

export async function humanize(
  inputText: string,
  mode: HumanizeMode,
  wordLimit: number
): Promise<string> {
  const truncated = truncateToWordLimit(inputText, wordLimit);

  // Split into variable-size chunks for natural local inconsistency
  const chunks = splitIntoVariableChunks(truncated);

  // Pass 1: Structural rewrite — parallel per chunk
  const structural = await structuralPass(chunks, mode);

  // Pass 2: Semantic naturalness — parallel per chunk (skipped for light)
  const semantic = await semanticPass(structural, mode);

  // Merge chunks back into full text
  const merged = semantic.join("\n\n");

  // Pass 3: Targeted mutation on full merged text (gated by score)
  return mutationPass(merged, mode);
}
