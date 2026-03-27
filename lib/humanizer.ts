import { GoogleGenerativeAI } from "@google/generative-ai";
import { getLightPrompt } from "@/prompts/light";
import { getMediumPrompt } from "@/prompts/medium";
import { getAggressivePrompt } from "@/prompts/aggressive";

export type HumanizeMode = "light" | "medium" | "aggressive";

function truncateToWordLimit(text: string, wordLimit: number): string {
  const words = text.trim().split(/\s+/);
  if (words.length <= wordLimit) return text.trim();
  return words.slice(0, wordLimit).join(" ");
}

function buildPrompt(text: string, mode: HumanizeMode): string {
  switch (mode) {
    case "light":
      return getLightPrompt(text);
    case "medium":
      return getMediumPrompt(text);
    case "aggressive":
      return getAggressivePrompt(text);
  }
}

async function humanizeWithGemini(prompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY not set");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-flash",
    generationConfig: {
      temperature: 0.85,
      topP: 0.95,
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

async function humanizeWithHuggingFace(prompt: string): Promise<string> {
  const apiKey = process.env.HUGGINGFACE_API_KEY;
  const HF_API_URL =
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2";

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  const body = JSON.stringify({
    inputs: `<s>[INST] ${prompt} [/INST]`,
    parameters: {
      max_new_tokens: 2048,
      temperature: 0.85,
      top_p: 0.95,
      do_sample: true,
      return_full_text: false,
    },
  });

  const response = await fetch(HF_API_URL, { method: "POST", headers, body });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`HuggingFace API error ${response.status}: ${err}`);
  }

  const data = await response.json();

  // HF returns array of generated objects
  if (Array.isArray(data) && data[0]?.generated_text) {
    return data[0].generated_text.trim();
  }

  throw new Error("Unexpected HuggingFace response format");
}

export async function humanize(
  inputText: string,
  mode: HumanizeMode,
  wordLimit: number
): Promise<string> {
  const truncated = truncateToWordLimit(inputText, wordLimit);
  const prompt = buildPrompt(truncated, mode);

  // Try Gemini first
  try {
    const result = await humanizeWithGemini(prompt);
    return result;
  } catch (geminiError) {
    console.warn("Gemini failed, falling back to HuggingFace:", geminiError);
  }

  // Fallback to HuggingFace
  try {
    const result = await humanizeWithHuggingFace(prompt);
    return result;
  } catch (hfError) {
    console.error("HuggingFace fallback also failed:", hfError);
    throw new Error(
      "Both Gemini and HuggingFace APIs failed. Please check your API keys and try again."
    );
  }
}
