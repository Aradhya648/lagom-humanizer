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

// Particle positions for animated hero background
const PARTICLES = [
  { top: "15%", left: "5%",  dur: 25, delay: 0  },
  { top: "25%", left: "12%", dur: 30, delay: 3  },
  { top: "8%",  left: "22%", dur: 22, delay: 7  },
  { top: "40%", left: "8%",  dur: 35, delay: 1  },
  { top: "60%", left: "15%", dur: 28, delay: 5  },
  { top: "75%", left: "5%",  dur: 32, delay: 11 },
  { top: "10%", left: "35%", dur: 27, delay: 4  },
  { top: "50%", left: "30%", dur: 24, delay: 9  },
  { top: "80%", left: "25%", dur: 38, delay: 2  },
  { top: "20%", left: "50%", dur: 29, delay: 6  },
  { top: "65%", left: "45%", dur: 33, delay: 13 },
  { top: "5%",  left: "60%", dur: 26, delay: 8  },
  { top: "35%", left: "65%", dur: 31, delay: 14 },
  { top: "55%", left: "70%", dur: 23, delay: 3  },
  { top: "85%", left: "60%", dur: 36, delay: 10 },
  { top: "15%", left: "78%", dur: 28, delay: 7  },
  { top: "42%", left: "85%", dur: 34, delay: 1  },
  { top: "70%", left: "80%", dur: 27, delay: 15 },
  { top: "90%", left: "90%", dur: 21, delay: 6  },
  { top: "30%", left: "95%", dur: 39, delay: 11 },
  { top: "8%",  left: "88%", dur: 25, delay: 4  },
  { top: "50%", left: "92%", dur: 30, delay: 9  },
  { top: "75%", left: "95%", dur: 35, delay: 0  },
  { top: "95%", left: "40%", dur: 22, delay: 12 },
] as const;

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

      {/* ── Header ── */}
      <header className="sticky top-0 z-20 bg-background/95 backdrop-blur-sm border-b border-border">
        <div className="max-w-6xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            {/* Techy sunflower — angular V-petals with terminal dots + crosshair center */}
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true">
              <circle cx="16" cy="16" r="2.8" stroke="#22D3A0" strokeWidth="1.1" />
              <line x1="12.8" y1="16" x2="19.2" y2="16" stroke="#22D3A0" strokeWidth="0.7" opacity="0.45"/>
              <line x1="16" y1="12.8" x2="16" y2="19.2" stroke="#22D3A0" strokeWidth="0.7" opacity="0.45"/>
              {[0, 45, 90, 135, 180, 225, 270, 315].map((angle) => (
                <g key={angle} transform={`rotate(${angle} 16 16)`} opacity={angle % 90 === 0 ? 1 : 0.55}>
                  <line x1="15.3" y1="13.2" x2="16" y2="4.2" stroke="#22D3A0" strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="16.7" y1="13.2" x2="16" y2="4.2" stroke="#22D3A0" strokeWidth="1.1" strokeLinecap="round"/>
                  <circle cx="16" cy="4.2" r="0.9" fill="#22D3A0" />
                </g>
              ))}
            </svg>
            <span className="font-serif italic font-bold text-[1.5rem] text-text leading-none tracking-tight">
              lagom
            </span>
            <span className="text-[0.95rem] font-sans font-light text-muted leading-none tracking-wide">
              Just the right amount of human
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-faint">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-opacity" />
            <span>Live</span>
          </div>
        </div>
      </header>

      {/* ── Hero Section ── */}
      <section className="relative overflow-hidden hero-bg">
        {/* Animated particles */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
          {PARTICLES.map((p, i) => (
            <span
              key={i}
              className={`particle particle-${i % 6}`}
              style={{
                top: p.top,
                left: p.left,
                animationDuration: `${p.dur}s`,
                animationDelay: `${p.delay}s`,
              }}
            />
          ))}
        </div>
        {/* Hero text */}
        <div className="relative z-10 max-w-6xl mx-auto px-5 sm:px-8 py-10 sm:py-16 text-center">
          <h1 className="font-serif italic gradient-text text-[2rem] sm:text-[3rem] leading-tight mb-3 tracking-tight">
            Humanize Your AI Text
          </h1>
          <p className="text-lg text-muted font-light max-w-md mx-auto leading-relaxed">
            Bypass AI detection and sound human every time
          </p>
        </div>
      </section>

      {/* ── Main ── */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-5 sm:px-8 py-6">

        {/* Controls Bar */}
        <div className="bg-surface border border-border rounded-2xl shadow-panel px-4 py-3 mb-5">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            {/* Content type */}
            <div className="flex-shrink-0">
              <ContentTypeSelector value={contentType} onChange={setContentType} />
            </div>

            {/* Word limit */}
            <div className="flex items-center gap-2.5 flex-1 min-w-0">
              <span className="text-xs text-faint whitespace-nowrap font-medium uppercase tracking-widest">
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
                style={deepMode
                  ? { boxShadow: "0 0 18px rgba(34,211,160,0.45), 0 0 6px rgba(34,211,160,0.3)" }
                  : { boxShadow: "0 0 10px rgba(34,211,160,0.12)" }}
                className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 ${
                  deepMode
                    ? "border-2 border-accent text-accent bg-accent-surface"
                    : "border border-border text-muted hover:border-accent/40 hover:text-accent/80 hover:bg-accent-surface/50"
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${deepMode ? "bg-accent animate-pulse-opacity" : "bg-faint/50"}`} />
                Deep Mode
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
          <div className="mb-5 px-4 py-3 rounded-xl bg-red-950/50 border border-red-800/50 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Two-panel layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

          {/* Input Panel */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between px-1">
              <span className="text-[11px] font-semibold text-accent/50 uppercase tracking-widest" style={{ textShadow: "0 0 8px rgba(0,212,170,0.4)" }}>
                Input
              </span>
              <div className="flex items-center gap-2">
                {inputScore && (
                  <ScorePill score={inputScore.score} label={inputScore.label} />
                )}
              </div>
            </div>
            <div className="bg-surface border border-border rounded-2xl shadow-panel overflow-hidden h-80 lg:h-[460px]">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste your AI-generated text here..."
                className="w-full h-full resize-none bg-transparent px-5 py-5 text-[15px] text-text placeholder:text-faint/50 leading-relaxed focus:outline-none"
                spellCheck={false}
              />
            </div>
            <div className="flex items-center justify-between px-1">
              <span className={`text-xs tabular-nums transition-colors ${isOverLimit ? "text-orange-400" : "text-faint"}`}>
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
              <span className="text-[11px] font-semibold text-accent/50 uppercase tracking-widest" style={{ textShadow: "0 0 8px rgba(0,212,170,0.4)" }}>
                Output
              </span>
              <div className="flex items-center gap-2">
                {inputScore && outputScore && (
                  <span className="text-xs text-faint tabular-nums flex items-center gap-1">
                    <span className={`font-medium ${getScoreColor(inputScore.score)}`}>
                      {inputScore.score}
                    </span>
                    <span className="text-faint/40">→</span>
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
                loading ? "border-accent/20 shadow-panel-active" : "border-border"
              }`}
            >
              {loading && (
                <div className="absolute inset-0 flex items-center justify-center p-6">
                  <div className="flex flex-col items-center gap-3 max-w-sm text-center">
                    <span className="text-accent">
                      <Spinner size={22} />
                    </span>
                    <span className="text-xs text-faint">
                      {deepMode && deepStatus ? deepStatus : `Rewriting as ${contentType}...`}
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
                <div className="px-5 py-5">
                  <span className="text-[15px] text-faint/50 leading-relaxed">
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
                      ? "border-emerald-700/50 text-emerald-400 bg-emerald-950/60"
                      : "border-border text-muted hover:border-accent/40 hover:text-accent hover:bg-accent-surface"
                  }`}
                >
                  {copied ? "Copied!" : "Copy"}
                </button>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-border py-5 mt-4">
        <div className="max-w-6xl mx-auto px-5 sm:px-8 flex items-center justify-between">
          <span className="text-[11px] tracking-widest uppercase font-medium text-accent/40 w-1/3">
            Lagom · Leverage AI Smartly
          </span>
          <span className="w-1/3 text-center text-[11px] font-bold tracking-widest uppercase gradient-text" style={{ filter: "drop-shadow(0 0 6px rgba(0,212,170,0.3))" }}>
            By · Drufiy AI Pvt. Ltd.
          </span>
          <span className="text-[11px] tracking-widest uppercase font-medium text-accent/40 w-1/3 text-right">
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
