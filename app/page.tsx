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
      <header className="sticky top-0 z-20 bg-background/95 backdrop-blur-sm border-b border-border">
        <div className="max-w-6xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3.5">
            <span className="font-serif italic text-[1.6rem] text-text leading-none tracking-tight">
              lagom
            </span>
            <span className="hidden sm:block text-faint text-xs tracking-wide pt-0.5 font-light">
              just the right amount of human
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-faint">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse-opacity" />
            <span>Live</span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-5 sm:px-8 py-7">

        {/* Controls Bar */}
        <div className="bg-surface border border-border rounded-2xl shadow-panel px-4 py-3 mb-5">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            {/* Content type selector */}
            <div className="flex-shrink-0">
              <ContentTypeSelector value={contentType} onChange={setContentType} />
            </div>

            {/* Word limit */}
            <div className="flex items-center gap-2.5 flex-1 min-w-0">
              <span className="text-xs text-faint whitespace-nowrap font-medium">
                Words
              </span>
              <input
                type="range"
                min={100}
                max={1000}
                step={50}
                value={wordLimit}
                onChange={(e) => setWordLimit(Number(e.target.value))}
                className="flex-1 max-w-28"
                aria-label="Word limit slider"
              />
              <span className="text-xs text-accent font-medium tabular-nums w-9 text-right">
                {wordLimit}
              </span>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2 w-full sm:w-auto">
              {/* Deep Mode toggle */}
              <button
                onClick={() => setDeepMode(d => !d)}
                title={deepMode ? "Deep Mode on — 4-detector loop, ~60–90s" : "Enable Deep Mode — scores all detectors and iterates"}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-xl border text-xs font-medium transition-all duration-200 ${
                  deepMode
                    ? "border-accent/40 text-accent bg-accent-surface"
                    : "border-border text-muted hover:border-accent/30 hover:text-accent/80 hover:bg-accent-surface/50"
                }`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${deepMode ? "bg-accent animate-pulse-opacity" : "bg-faint/50"}`} />
                Deep
              </button>

              {/* Humanize button */}
              <button
                onClick={handleHumanize}
                disabled={loading || !inputText.trim()}
                className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2 rounded-xl bg-accent text-background font-semibold text-sm shadow-btn transition-all duration-200 hover:bg-accent-dim disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
              >
                {loading ? (
                  <>
                    <Spinner size={14} />
                    <span>{deepMode ? "Analyzing..." : "Rewriting..."}</span>
                  </>
                ) : (
                  deepMode ? "Deep Humanize" : "Humanize"
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-5 px-4 py-3 rounded-xl bg-red-950/50 border border-red-800/60 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Two-panel layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

          {/* Input Panel */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between px-1">
              <span className="text-[11px] font-semibold text-faint uppercase tracking-widest">
                Input
              </span>
              <div className="flex items-center gap-2">
                {inputScore && (
                  <ScorePill score={inputScore.score} label={inputScore.label} />
                )}
              </div>
            </div>
            <div className="bg-surface border border-border rounded-2xl shadow-panel overflow-hidden">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste your AI-generated text here..."
                className="w-full h-80 lg:h-[460px] resize-none bg-transparent px-5 py-5 text-[15px] text-text placeholder:text-faint/60 leading-relaxed focus:outline-none"
                spellCheck={false}
              />
            </div>
            <div className="flex items-center justify-between px-1">
              <span
                className={`text-xs tabular-nums transition-colors ${
                  isOverLimit ? "text-orange-400" : "text-faint"
                }`}
              >
                {wordCount.toLocaleString()} / {MAX_WORDS.toLocaleString()} words
              </span>
              {isOverLimit && (
                <span className="text-xs text-orange-400">
                  First {wordLimit} words only
                </span>
              )}
            </div>
          </div>

          {/* Output Panel */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between px-1">
              <span className="text-[11px] font-semibold text-faint uppercase tracking-widest">
                Output
              </span>
              <div className="flex items-center gap-2">
                {inputScore && outputScore && (
                  <span className="text-xs text-faint tabular-nums flex items-center gap-1">
                    <span className={`font-medium ${getScoreColor(inputScore.score)}`}>
                      {inputScore.score}
                    </span>
                    <span className="text-faint/50">→</span>
                    <span className={`font-medium ${getScoreColor(outputScore.score)}`}>
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
              className={`relative w-full h-80 lg:h-[460px] bg-surface border rounded-2xl shadow-panel overflow-y-auto transition-all duration-300 ${
                loading ? "border-accent/25" : "border-border"
              }`}
            >
              {loading && (
                <div className="absolute inset-0 flex items-center justify-center p-6">
                  <div className="flex flex-col items-center gap-3 text-muted max-w-sm text-center">
                    <Spinner size={22} />
                    <span className="text-xs text-faint">
                      {deepMode && deepStatus
                        ? deepStatus
                        : `Rewriting as ${contentType}...`}
                    </span>
                    {deepMode && liveScores && (
                      <div className="flex flex-wrap items-center justify-center gap-3 pt-1">
                        {(["gptzero", "zerogpt", "quillbot", "originality"] as const).map(k => {
                          const s = liveScores[k];
                          if (s === -1) return null;
                          const color = s <= 15 ? "text-emerald-400" : s <= 40 ? "text-amber-400" : "text-red-400";
                          const labels: Record<string, string> = { gptzero: "GPTZero", zerogpt: "ZeroGPT", quillbot: "QuillBot", originality: "Origin." };
                          return (
                            <span key={k} className="text-[10px] tabular-nums flex items-center gap-1">
                              <span className="text-faint/70">{labels[k]}</span>
                              <span className={`font-semibold ${color}`}>{s}%</span>
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
                  <span className="text-sm text-faint/60 text-center px-10 leading-relaxed">
                    Your humanized text will appear here
                  </span>
                </div>
              )}
              {!loading && outputText && (
                <div className="px-5 py-5">
                  <WordReveal text={outputText} />
                </div>
              )}
            </div>

            {/* Deep mode detector scores */}
            {deepScores && (
              <div className="flex flex-wrap items-center gap-3 px-1">
                {deepIterations != null && (
                  <span className="text-xs text-faint">
                    {deepIterations} iter{deepIterations !== 1 ? "s" : ""}
                  </span>
                )}
                {(["gptzero", "zerogpt", "quillbot", "originality"] as const).map(k => {
                  const s = deepScores[k];
                  if (s === -1) return null;
                  const color = s <= 15 ? "text-emerald-400" : s <= 40 ? "text-amber-400" : "text-red-400";
                  const labels: Record<string, string> = { gptzero: "GPTZero", zerogpt: "ZeroGPT", quillbot: "QuillBot", originality: "Origin." };
                  return (
                    <span key={k} className="text-xs tabular-nums flex items-center gap-1">
                      <span className="text-faint/70">{labels[k]}</span>
                      <span className={`font-semibold ${color}`}>{s}%</span>
                    </span>
                  );
                })}
              </div>
            )}

            <div className="flex items-center justify-between px-1">
              <span className="text-xs text-faint tabular-nums">
                {outputWordCount > 0 ? `${outputWordCount.toLocaleString()} words` : ""}
              </span>
              {outputText && (
                <button
                  onClick={handleCopy}
                  className={`text-xs px-3 py-1.5 rounded-lg border transition-all duration-200 font-medium ${
                    copied
                      ? "border-emerald-800/60 text-emerald-400 bg-emerald-950/50"
                      : "border-border text-muted hover:border-accent/40 hover:text-accent hover:bg-accent-surface/50"
                  }`}
                >
                  {copied ? "Copied!" : "Copy"}
                </button>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-5 mt-4">
        <div className="max-w-6xl mx-auto px-5 sm:px-8 flex items-center justify-between">
          <span className="text-xs text-faint">
            <span className="font-serif italic text-muted">Lagom</span>
            {" · "}Built for writers, not robots
          </span>
          <span className="text-xs text-faint/60">
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
  if (score <= 60) return "text-amber-400";
  if (score <= 80) return "text-orange-400";
  return "text-red-400";
}
