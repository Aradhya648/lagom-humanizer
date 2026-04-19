"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import ScorePill from "@/components/ScorePill";
import ContentTypeSelector, { ContentType } from "@/components/ContentTypeSelector";
import WordReveal from "@/components/WordReveal";
import Spinner from "@/components/Spinner";
import { detectAI, DetectionResult } from "@/lib/detector";

const MAX_WORDS = 1000;
// If NEXT_PUBLIC_DEEP_API_URL is set, deep calls go cross-domain (Vercel → Railway).
// If unset (i.e. running ON Railway), deep calls go to same origin via relative path.
const DEEP_API_URL = process.env.NEXT_PUBLIC_DEEP_API_URL ?? "";

function countWords(text: string): number {
  return text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
}

interface DeepScores {
  gptzero: number;
  zerogpt: number;
  quillbot: number;
  originality: number;
}

export default function Home() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [contentType, setContentType] = useState<ContentType>("general");
  const [wordLimit, setWordLimit] = useState(1000);
  const [loading, setLoading] = useState(false);
  const [deepMode, setDeepMode] = useState(false);
  const [deepScores, setDeepScores] = useState<DeepScores | null>(null);
  const [deepIterations, setDeepIterations] = useState<number | null>(null);
  const [deepStatus, setDeepStatus] = useState<string>("");
  const [liveScores, setLiveScores] = useState<DeepScores | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const [inputScore, setInputScore] = useState<DetectionResult | null>(null);
  const [outputScore, setOutputScore] = useState<DetectionResult | null>(null);

  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Live AI detection on input with 600ms debounce
  useEffect(() => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current);

    if (inputText.trim().length === 0) {
      setInputScore(null);
      return;
    }

    debounceTimer.current = setTimeout(() => {
      const result = detectAI(inputText);
      setInputScore(result);
    }, 600);

    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
    };
  }, [inputText]);

  const handleHumanize = useCallback(async () => {
    if (!inputText.trim() || loading) return;
    setLoading(true);
    setError(null);
    setOutputText("");
    setOutputScore(null);
    setDeepScores(null);
    setDeepIterations(null);
    setDeepStatus("");
    setLiveScores(null);

    try {
      if (deepMode) {
        const deepEndpoint = `${DEEP_API_URL}/api/humanize-deep`;
        const res = await fetch(deepEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: inputText, contentType, wordLimit }),
        });

        if (!res.ok) {
          const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
          throw new Error(data.error || "Deep humanize failed");
        }
        if (!res.body) throw new Error("No response stream");

        // Read SSE stream line-by-line
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finalResult: { text: string; iterations: number; finalScores: { gptzero?: { score: number }; zerogpt?: { score: number }; quillbot?: { score: number }; originality?: { score: number } } } | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() ?? "";

          for (const part of parts) {
            const lines = part.split("\n");
            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const json = line.slice(6).trim();
              if (!json) continue;
              try {
                const evt = JSON.parse(json);
                if (evt.type === "status") {
                  setDeepStatus(evt.message);
                } else if (evt.type === "score") {
                  setLiveScores({
                    gptzero: evt.scores?.gptzero?.score ?? -1,
                    zerogpt: evt.scores?.zerogpt?.score ?? -1,
                    quillbot: evt.scores?.quillbot?.score ?? -1,
                    originality: evt.scores?.originality?.score ?? -1,
                  });
                } else if (evt.type === "result") {
                  finalResult = evt;
                } else if (evt.type === "error") {
                  throw new Error(evt.message);
                }
              } catch (parseErr) {
                // Swallow JSON parse errors and Safari DOMExceptions from
                // incomplete SSE frames — only rethrow app-level errors
                // (i.e. those we explicitly threw from evt.type === "error").
                const msg = parseErr instanceof Error ? parseErr.message : "";
                const isAppError = msg && !msg.includes("JSON") && !msg.includes("token") && !msg.includes("expected pattern") && !msg.includes("Unexpected end");
                if (isAppError) throw parseErr;
              }
            }
          }
        }

        if (!finalResult) throw new Error("Deep mode requires a backend with Playwright. Make sure NEXT_PUBLIC_DEEP_API_URL points to your Railway deployment.");

        setOutputText(finalResult.text);
        setDeepIterations(finalResult.iterations);
        setDeepStatus("");

        const fs = finalResult.finalScores;
        setDeepScores({
          gptzero: fs?.gptzero?.score ?? -1,
          zerogpt: fs?.zerogpt?.score ?? -1,
          quillbot: fs?.quillbot?.score ?? -1,
          originality: fs?.originality?.score ?? -1,
        });

        const { detectAI } = await import("@/lib/detector");
        const r = detectAI(finalResult.text);
        setOutputScore(r);
        const orig = detectAI(inputText);
        setInputScore(orig);
      } else {
        // Fast mode — Vercel endpoint
        const res = await fetch("/api/humanize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: inputText, contentType, wordLimit }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Something went wrong");

        setOutputText(data.humanizedText);
        setOutputScore({ score: data.humanizedScore, label: getLabelFromScore(data.humanizedScore) });
        setInputScore({ score: data.originalScore, label: getLabelFromScore(data.originalScore) });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  }, [inputText, contentType, wordLimit, loading, deepMode]);

  const handleCopy = useCallback(async () => {
    if (!outputText) return;
    try {
      await navigator.clipboard.writeText(outputText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback
    }
  }, [outputText]);

  const wordCount = countWords(inputText);
  const outputWordCount = countWords(outputText);
  const isOverLimit = wordCount > MAX_WORDS;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-border bg-background/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="font-serif italic text-2xl text-text leading-none tracking-tight">
              lagom
            </span>
            <span className="hidden sm:block text-muted text-xs tracking-wide pt-0.5">
              just the right amount of human
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs text-muted">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Live
            </span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
        {/* Controls Bar */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 mb-6">
          <div className="w-full sm:w-auto sm:min-w-72">
            <ContentTypeSelector value={contentType} onChange={setContentType} />
          </div>
          <div className="flex items-center gap-3 flex-1">
            <span className="text-xs text-muted whitespace-nowrap">
              Word limit
            </span>
            <input
              type="range"
              min={100}
              max={1000}
              step={50}
              value={wordLimit}
              onChange={(e) => setWordLimit(Number(e.target.value))}
              className="flex-1 max-w-32"
              aria-label="Word limit slider"
            />
            <span className="text-xs text-accent font-medium tabular-nums w-10">
              {wordLimit}
            </span>
          </div>
          <div className="flex items-center gap-3 w-full sm:w-auto">
            {/* Deep Mode toggle — always visible */}
            <button
              onClick={() => setDeepMode(d => !d)}
              title={deepMode ? "Deep Mode on — 4-detector loop, ~60–90s" : "Enable Deep Mode — scores all detectors and iterates"}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-xl border text-xs font-medium transition-all duration-200 ${
                deepMode
                  ? "border-accent/60 text-accent bg-accent/10"
                  : "border-border text-muted hover:border-accent/30 hover:text-accent/70"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${deepMode ? "bg-accent animate-pulse" : "bg-muted/40"}`} />
              Deep
            </button>
            <button
              onClick={handleHumanize}
              disabled={loading || !inputText.trim()}
              className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-accent text-background font-semibold text-sm transition-all duration-200 hover:bg-accent-dim disabled:opacity-40 disabled:cursor-not-allowed active:scale-95"
            >
              {loading ? (
                <>
                  <Spinner size={15} />
                  {deepMode ? "Deep analyzing..." : "Humanizing..."}
                </>
              ) : (
                deepMode ? "Deep Humanize" : "Humanize"
              )}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 px-4 py-3 rounded-xl bg-red-950 border border-red-800 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Two-panel layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Input Panel */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted uppercase tracking-wider">
                Input
              </span>
              <div className="flex items-center gap-2">
                {inputScore && (
                  <ScorePill score={inputScore.score} label={inputScore.label} />
                )}
              </div>
            </div>
            <div className="relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste your AI-generated text here..."
                className="w-full h-80 lg:h-[460px] resize-none bg-surface border border-border rounded-2xl px-4 py-4 text-sm text-text placeholder:text-muted/50 leading-relaxed transition-all duration-200 focus:border-accent/40"
                spellCheck={false}
              />
            </div>
            <div className="flex items-center justify-between px-1">
              <span
                className={`text-xs tabular-nums transition-colors ${
                  isOverLimit ? "text-orange-400" : "text-muted"
                }`}
              >
                {wordCount.toLocaleString()} / {MAX_WORDS.toLocaleString()} words
              </span>
              {isOverLimit && (
                <span className="text-xs text-orange-400">
                  Only the first {wordLimit} words will be processed
                </span>
              )}
            </div>
          </div>

          {/* Output Panel */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted uppercase tracking-wider">
                Output
              </span>
              <div className="flex items-center gap-2">
                {inputScore && outputScore && (
                  <span className="text-xs text-muted tabular-nums flex items-center gap-1">
                    <span
                      className={`font-medium ${getScoreColor(inputScore.score)}`}
                    >
                      {inputScore.score}
                    </span>
                    <span>→</span>
                    <span
                      className={`font-medium ${getScoreColor(outputScore.score)}`}
                    >
                      {outputScore.score}
                    </span>
                    {outputScore.score < inputScore.score && (
                      <span className="text-emerald-400">✓</span>
                    )}
                  </span>
                )}
                {outputScore && (
                  <ScorePill score={outputScore.score} label={outputScore.label} />
                )}
              </div>
            </div>

            <div
              className={`relative w-full h-80 lg:h-[460px] bg-surface border rounded-2xl px-4 py-4 overflow-y-auto transition-all duration-300 ${
                loading
                  ? "border-accent/30"
                  : outputText
                  ? "border-border"
                  : "border-border"
              }`}
            >
              {loading && (
                <div className="absolute inset-0 flex items-center justify-center p-6">
                  <div className="flex flex-col items-center gap-3 text-muted max-w-sm text-center">
                    <Spinner size={24} />
                    <span className="text-xs">
                      {deepMode && deepStatus
                        ? deepStatus
                        : `Rewriting as ${contentType}...`}
                    </span>
                    {deepMode && liveScores && (
                      <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                        {(["gptzero", "zerogpt", "quillbot", "originality"] as const).map(k => {
                          const s = liveScores[k];
                          if (s === -1) return null;
                          const color = s <= 15 ? "text-emerald-400" : s <= 40 ? "text-yellow-400" : "text-red-400";
                          const labels: Record<string, string> = { gptzero: "GPTZero", zerogpt: "ZeroGPT", quillbot: "QuillBot", originality: "Origin." };
                          return (
                            <span key={k} className="text-[10px] tabular-nums flex items-center gap-1">
                              <span className="text-muted/60">{labels[k]}</span>
                              <span className={`font-medium ${color}`}>{s}%</span>
                            </span>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              )}
              {!loading && !outputText && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-sm text-muted/40 text-center px-8">
                    Your humanized text will appear here
                  </span>
                </div>
              )}
              {!loading && outputText && <WordReveal text={outputText} />}
            </div>

            {/* Deep mode detector scores */}
            {deepScores && (
              <div className="flex flex-wrap items-center gap-2 px-1 pb-1">
                <span className="text-xs text-muted">
                  {deepIterations != null ? `${deepIterations} iter${deepIterations !== 1 ? "s" : ""}` : ""}
                </span>
                {(["gptzero", "zerogpt", "quillbot", "originality"] as const).map(k => {
                  const s = deepScores[k];
                  if (s === -1) return null;
                  const color = s <= 15 ? "text-emerald-400" : s <= 40 ? "text-yellow-400" : "text-red-400";
                  const labels: Record<string, string> = { gptzero: "GPTZero", zerogpt: "ZeroGPT", quillbot: "QuillBot", originality: "Origin." };
                  return (
                    <span key={k} className="text-xs tabular-nums flex items-center gap-1">
                      <span className="text-muted/60">{labels[k]}</span>
                      <span className={`font-medium ${color}`}>{s}%</span>
                    </span>
                  );
                })}
              </div>
            )}
            <div className="flex items-center justify-between px-1">
              <span className="text-xs text-muted tabular-nums">
                {outputWordCount > 0
                  ? `${outputWordCount.toLocaleString()} words`
                  : ""}
              </span>
              {outputText && (
                <button
                  onClick={handleCopy}
                  className={`text-xs px-3 py-1.5 rounded-lg border transition-all duration-200 active:scale-95 ${
                    copied
                      ? "border-emerald-700 text-emerald-400 bg-emerald-950"
                      : "border-border text-muted hover:border-accent/50 hover:text-accent"
                  }`}
                >
                  {copied ? "Copied!" : "Copy to clipboard"}
                </button>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between">
          <span className="text-xs text-muted">
            <span className="font-serif italic text-text/60">Lagom</span>
            {" · "}Built for writers, not robots
          </span>
          <span className="text-xs text-muted/40">
            Free to use
          </span>
        </div>
      </footer>
    </div>
  );
}

function getLabelFromScore(score: number): DetectionResult["label"] {
  if (score <= 30) return "Likely Human";
  if (score <= 60) return "Mixed";
  if (score <= 80) return "Likely AI";
  return "Almost Certainly AI";
}

function getScoreColor(score: number): string {
  if (score <= 30) return "text-emerald-400";
  if (score <= 60) return "text-yellow-400";
  if (score <= 80) return "text-orange-400";
  return "text-red-400";
}
