"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import ScorePill from "@/components/ScorePill";
import ContentTypeSelector, { ContentType } from "@/components/ContentTypeSelector";
import WordReveal from "@/components/WordReveal";
import Spinner from "@/components/Spinner";
import { detectAI, DetectionResult } from "@/lib/detector";

const MAX_WORDS = 1000;

function countWords(text: string): number {
  return text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
}

export default function Home() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [contentType, setContentType] = useState<ContentType>("general");
  const [wordLimit, setWordLimit] = useState(1000);
  const [loading, setLoading] = useState(false);
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

    try {
      const res = await fetch("/api/humanize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText, contentType, wordLimit }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Something went wrong");
      }

      setOutputText(data.humanizedText);
      setOutputScore({ score: data.humanizedScore, label: getLabelFromScore(data.humanizedScore) });
      // Also update input score from API response
      setInputScore({ score: data.originalScore, label: getLabelFromScore(data.originalScore) });
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  }, [inputText, contentType, wordLimit, loading]);

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
          <button
            onClick={handleHumanize}
            disabled={loading || !inputText.trim()}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-accent text-background font-semibold text-sm transition-all duration-200 hover:bg-accent-dim disabled:opacity-40 disabled:cursor-not-allowed active:scale-95"
          >
            {loading ? (
              <>
                <Spinner size={15} />
                Humanizing...
              </>
            ) : (
              "Humanize"
            )}
          </button>
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
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="flex flex-col items-center gap-3 text-muted">
                    <Spinner size={24} />
                    <span className="text-xs">Rewriting as {contentType}...</span>
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
