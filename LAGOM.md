# LAGOM — Single Source of Truth

> *Just the right amount of human.*

**Live app:** https://lagom-one.vercel.app
**Repo:** https://github.com/Aradhya648/lagom-humanizer
**Owner:** Aradhya Mishra (Drufiy / lagom-humanizer)
**Last updated:** 2026-07-11

---

## Table of Contents

1. [Goals](#goals)
2. [Tech Stack & Deployment](#tech-stack--deployment)
3. [Architecture Overview](#architecture-overview)
4. [Fast Mode Pipeline (in detail)](#fast-mode-pipeline-in-detail)
5. [Deep Mode Pipeline](#deep-mode-pipeline)
6. [Deterministic Pass Stack](#deterministic-pass-stack)
7. [Environment Variables](#environment-variables)
8. [What We Tried & Why It Failed](#what-we-tried--why-it-failed)
9. [Peak Scores Achieved](#peak-scores-achieved)
10. [Current State (2026-07-11)](#current-state-2026-07-11)
11. [Roadmap](#roadmap)
12. [Session Log](#session-log)

---

## Goals

### Short-term
- **0% AI** on lightweight detectors: QuillBot, ZeroGPT

### Long-term
- **100% Human** on heavyweight detectors: GPTZero, Originality.ai, Turnitin, DrillBit

---

## Tech Stack & Deployment

| Layer | Technology |
|-------|------------|
| Framework | Next.js 16 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS (aurora/northern-lights theme) |
| Primary LLM | Google Gemini 2.5 Flash |
| Frontend deploy | Vercel (lagom-one.vercel.app) |
| Backend deploy | Railway (Playwright/Chromium for deep mode scrapers) |
| Email alerts | Resend (drufiyai001@gmail.com) |

### Deployment Notes
- **Vercel** hosts the Next.js app. Auto-deploy from GitHub is currently broken — deploy via `vercel --prod` CLI.
- **Railway** hosts the deep-mode API (`/api/humanize-deep`) which requires Playwright/Chromium for live detector scraping.
- `GEMINI_API_KEY` must be set in both Vercel and Railway environment variables.
- `NVIDIA_API_KEY` must NOT be active on Vercel — see [What We Tried](#what-we-tried--why-it-failed).

---

## Architecture Overview

```
User (browser)
   │
   ├── Types text → live heuristic detector (client-side, 600ms debounce)
   │
   └── Clicks "Humanize" → POST /api/humanize
                               │
                          [Fast Mode — Vercel serverless, 60s limit]
                               │
                          lib/humanizer.ts → humanize()
                               │
                          ┌────▼─────────────────────────────────┐
                          │  GLORY-DAYS PIPELINE                 │
                          │  1. splitIntoVariableChunks()        │
                          │  2. structuralPass()   (parallel)    │
                          │  3. semanticPass()     (parallel)    │
                          │  4. mutationPass()     (gated)       │
                          │  5. finishHumanizedText()            │
                          │     └── 10 deterministic passes      │
                          └──────────────────────────────────────┘
                               │
                          Returns { humanizedText, originalScore, humanizedScore }
                               │
                          [Deep Mode — Railway, 6min limit]
                               │
                          lib/deep-humanizer.ts → deepHumanize()
                               │
                          Iterative loop (up to 4 rounds):
                          1. humanize() baseline
                          2. scoreAllDetectors() via Playwright
                          3. Surgery on flagged sentences
                          4. roundTripHumanize() nuclear reset (if stuck)
                          5. Emit SSE events to client
```

---

## Fast Mode Pipeline (in detail)

All code lives in `lib/humanizer.ts` (~1,570 lines).

### Entry: `humanize(text, contentType, wordLimit)`

```
Input text
  │
  ├── truncateToWordLimit() — enforce word limit
  ├── classifyRegister() — academic / formal / neutral / informal
  │
  ├── [<30 words] → singlePassHumanize() → finishHumanizedText()
  │
  └── [≥30 words] → GLORY-DAYS PIPELINE:
        │
        ├── splitIntoVariableChunks()
        │     Groups paragraphs into chunks ≤250 words each.
        │     Variable sizing prevents uniform chunk-edge artifacts.
        │
        ├── Pass 1 — structuralPass() [parallel per chunk]
        │     Gemini 2.5 Flash, temp 0.60
        │     Restructures sentences: reorders, splits, merges.
        │     Prompt: getStructuralPrompt() from prompts/pipeline.ts
        │
        ├── Pass 2 — semanticPass() [parallel per chunk]
        │     Gemini 2.5 Flash, temp 0.55
        │     Word-level naturalness: plain vocab, no AI buzzwords.
        │     Prompt: getSemanticPrompt() from prompts/pipeline.ts
        │
        ├── [DISABLED] nvidiaFingerprintBreakPass()
        │     Was: gpt-oss-20b between semantic and mutation.
        │     REGRESSED scores — see "What We Tried" section.
        │     Silently skipped when NVIDIA_API_KEY is absent.
        │
        ├── Pass 3 — mutationPass() [gated — only if midScore > 25]
        │     Gemini 2.5 Flash, temp 0.60
        │     Targeted rewrite of still-flagged regions.
        │     Gating prevents over-paraphrasing on already-clean text.
        │
        └── finishHumanizedText()
              └── 10 deterministic passes (see below)
              └── enforceLengthDiscipline() (92% floor, 112% ceiling)
              └── iterative expansion if under floor (up to 2 attempts)
```

### Register Classifier (`classifyRegister`)
Reads the *input* text to determine register, which is used as a ceiling by deterministic passes so formal essays don't get colloquialized:
- `academic` — formal vocabulary, long sentences, academic markers
- `formal` — professional, no contractions, formal markers
- `neutral` — everyday professional
- `informal` — contractions, conversational language

### Model Config
```
FAST_MODEL = "gemini-2.5-flash"
STRUCTURAL_SETTINGS = { temperature: 0.60, topP: 0.90, maxOutputTokens: 4096 }
SEMANTIC_SETTINGS   = { temperature: 0.55, topP: 0.88, maxOutputTokens: 4096 }
MUTATION_SETTINGS   = { temperature: 0.60, topP: 0.88, maxOutputTokens: 4096 }
GEMINI_TIMEOUT_MS   = 30_000
```

---

## Deep Mode Pipeline

All code in `lib/deep-humanizer.ts`. Runs on Railway (requires Playwright/Chromium).

```
humanize() baseline
      │
      └── Loop (max 4 iterations):
            │
            ├── scoreAllDetectors() — live Playwright scrapes:
            │     QuillBot  → playwright headless
            │     ZeroGPT   → playwright headless
            │     GPTZero   → playwright headless
            │     Writer.com → playwright headless
            │
            ├── If all scores < 20% → done ✓
            │
            ├── Surgery mode:
            │     - Extract highlighted/flagged sentences from detector HTML
            │     - Rewrite ONLY those sentences via Gemini
            │     - Splice back into full text
            │     - Apply full 10-pass deterministic stack
            │
            └── Nuclear reset (if surgery doesn't budge score):
                  roundTripHumanize() — translate to Chinese, back to English
                  → Apply full 10-pass deterministic stack
```

Session slots: `MAX_DEEP_SESSIONS = 1` (Railway RAM constraint — parallel Chromium instances cause OOM).

---

## Deterministic Pass Stack

Applied in `finishHumanizedText()` in this exact order. Each pass is pure (no LLM calls).

| # | Pass | Purpose | Key Targets |
|---|------|---------|-------------|
| 1 | `antiPatternPass` | Kill AI paragraph openers | "Furthermore,", "Moreover,", "In conclusion,", "It is worth noting" |
| 2 | `rhetoricalSuppressionPass` | Strip connector/fluency overload | Overuse of "however", "therefore", "nevertheless" |
| 3 | `patternKillerPass` | Remove LLM fingerprint phrases | "X, Y, and Z alike", "The evidence is clear.", tautological pairs |
| 4 | `zeroGPTNgramBreaker` | Disrupt AI n-gram patterns | 30 hardcoded n-gram rewrite rules ZeroGPT flags |
| 5 | `structuralPerplexityInjector` | (currently a no-op) | Modes A, B, and now C all disabled — see below |
| 6 | `perplexityInjector` | Rare synonym injection | Up to 3 per paragraph, register-aware |
| 7 | `interParagraphDivergencePass` | Cross-paragraph vocab variance | Prevents same-word repetition across paragraphs |
| 8 | `burstinessInjector` | Sentence-length variance | Strategy A only: split at a comma before and/but/or/yet/so, but ONLY if it's the sentence's sole comma (rules out Oxford-comma list items) |
| 9 | `openerDiversityPass` | No duplicate sentence-opening words | Swaps 2nd+ occurrence in same paragraph; swap table is number-agreement-safe |
| 10 | `grammarAndDedupPass` | Cleanup + deduplication | Capitalizes sentence-starts (new safety net), fixes grammar artifacts, removes duplicate sentences |

### Disabled Passes (do not re-enable without benchmarking)
- `emDashInjector` — proven to REGRESS ZeroGPT (8%→24%) and QuillBot (16%→62%)
- `structuralPerplexityInjector` — ALL THREE modes now disabled. A & B: our own phrases "(admittedly)", "(broadly speaking)" were being flagged as lagom-humanizer signatures. C (comma→semicolon): confirmed 2026-07-11 to produce ungrammatical semicolons before dependent clauses ("...must-have; which will take...") — function is now a pure no-op, kept only for import compatibility.
- `burstinessInjector` Strategy B — "This distinction matters.", "The evidence is clear." bridge phrases were flagged

---

## Environment Variables

| Variable | Required | Where | Notes |
|----------|----------|-------|-------|
| `GEMINI_API_KEY` | Yes | Vercel + Railway | Google AI Studio key |
| `HUGGINGFACE_API_KEY` | Optional | — | Not currently used in main pipeline |
| `NVIDIA_API_KEY` | **DO NOT SET** | Vercel | Activates gpt-oss-20b pass — regresses QuillBot/ZeroGPT |
| `NVIDIA_MODEL` | — | — | Override model if NVIDIA pass ever reactivated |
| `RESEND_API_KEY` | Yes | Vercel | Error email alerts to drufiyai001@gmail.com |
| `NEXT_PUBLIC_DEEP_API_URL` | Yes | Vercel | Railway URL for deep-mode API |

---

## What We Tried & Why It Failed

> This section exists so we don't re-litigate decisions that cost sessions to reach.

### ❌ emDashInjector
**What:** Inject em-dashes (X — Y) as a "human punctuation" signal.
**Why it failed:** GPTZero doesn't see many em-dashes in its AI training data, so it briefly helped. ZeroGPT and QuillBot flagged them as synthetic after pattern-matching learned the exact injection templates. Observed: ZeroGPT 8%→24%, QuillBot 16%→62%.
**Status:** Disabled permanently in `finishHumanizedText`. Do not re-enable.

### ❌ structuralPerplexityInjector Modes A & B
**What:** Mode A inserts parenthetical asides "(admittedly)", "(broadly speaking)". Mode B injects em-dash list breaks.
**Why it failed:** Detectors learned these were lagom-humanizer signatures. The exact strings were highlighted in detector output as AI evidence.
**Status:** Only Mode C (comma→semicolon) is active. Do not re-enable A or B.

### ❌ burstinessInjector Strategy B
**What:** Injects bridge sentences: "This distinction matters.", "The evidence is clear."
**Why it failed:** Same signature problem — detectors flagged these canned bridges as evidence of AI authorship.
**Status:** Only Strategy A (sentence-split at comma) is active.

### ❌ NVIDIA gpt-oss-20b fingerprint break pass
**What:** Added April 24, 2026 (commit c885680). Passed merged Gemini output through openai/gpt-oss-20b (NVIDIA NIM free API) as a fingerprint-break mid-chain step.
**Why it failed:** gpt-oss-20b is GPT-lineage. QuillBot and ZeroGPT specialize in detecting ChatGPT-family statistical patterns. Swapped Gemini fingerprint (tolerated) for GPT fingerprint (heavily targeted). QuillBot: 6%→78%, ZeroGPT: 8%→57.8%.
**Status:** Pass still exists in code but is silently skipped when NVIDIA_API_KEY is absent. Do NOT add NVIDIA_API_KEY to Vercel environment.

### ❌ Round-trip translation (fast mode primary path)
**What:** Translate to Chinese, back to English, as the main humanization strategy (not just deep mode).
**Why it failed:** The back-translation produced polished academic prose — which re-introduced the exact AI-sounding sentence structures the detectors flag. Deep mode (which applies all 10 deterministic passes after translation) works fine; fast mode didn't have the passes to rough it up.
**Status:** Demoted to deep-mode-only nuclear reset. Fast mode uses the glory-days 3-LLM pipeline.

### ❌ 40+ prompt engineering iterations (milestones 1–32)
**What:** Over 30 milestones of prompt changes — adding instructions, banning words, restructuring the system prompt.
**Why it failed:** Diminishing returns after milestone ~15. Gemini's statistical signature cannot be removed via prompting — it's in the base model weights, not the instructions.
**Status:** Prompts are stable. Do not invest further sessions in prompt tweaks.

### ❌ Gemini 2.5 Pro for mutation pass
**What:** Used stronger Pro model for Pass 3.
**Why it failed:** Pro uses extended thinking tokens that truncate output mid-sentence at the 4096 token limit.
**Status:** All 3 passes use gemini-2.5-flash. Do not switch mutation back to Pro.

---

## Peak Scores Achieved

Achieved at milestone-41 (commit `1007a90`) and confirmed by the restored glory-days pipeline (commit `25154b6`):

| Detector | Fast Mode | Deep Mode |
|----------|-----------|-----------|
| QuillBot | ~0% | ~0% |
| ZeroGPT | 0% | 0% |
| GPTZero | 99% Human | ~80%+ Human |
| Originality.ai | unknown | unknown |

> These were measured manually on essay-length inputs (~400 words). No automated benchmark harness exists yet.

---

## Current State (2026-07-11)

### What's broken
- **GitHub→Vercel auto-deploy is broken** — deploy requires `vercel --prod` CLI after every push
- **No benchmark harness** — all testing is manual; no ground-truth spreadsheet
- **QuillBot still not confirmed fixed on real detectors** — internal heuristic score improved after the 2026-07-11 grammar-bug fixes, but not yet re-verified against live QuillBot/ZeroGPT

### What's working
- Full fast-mode pipeline operational
- Deep-mode pipeline operational on Railway
- **NVIDIA_API_KEY removed from Vercel** (2026-07-11) — gpt-oss-20b pass no longer active
- All 10 deterministic passes active; Mode C of structuralPerplexityInjector now disabled (grammar bug)
- 4 additional grammar/structure bugs fixed 2026-07-11 (see Session Log)
- Mobile UI, T&C page, error alerts, beta label — all live
- Real UI restored after Prash CI sabotage (commits 6c6c5af, 25154b6)

---

## Roadmap

### Week 1 (July 2026)

#### Day 1 — Stop the bleeding + establish ground truth
- [ ] Unset `NVIDIA_API_KEY` in Vercel dashboard (zero code change)
- [ ] Redeploy + manually verify QuillBot/ZeroGPT drop back to single digits
- [ ] Build benchmark: 10 essays × 4 detectors × spreadsheet
      (6 types: academic short, academic long, business, casual, technical, creative)
      This becomes the permanent ground truth — never discard or reset it

#### Day 2 — Ablation study
- [ ] Test 3 variants against all 10 benchmark essays:
  - **A:** Full deterministic stack (current, minus NVIDIA)
  - **B:** Only LLM passes, no deterministic layer
  - **C:** LLM passes + patternKillerPass + grammarAndDedupPass only
- [ ] Keep whichever wins. Record results in benchmark spreadsheet.

#### Day 3 — Fix the winner's known issues
- [ ] Fix any remaining grammar artifacts uncovered by benchmarks
- [ ] Re-baseline benchmark with fixes applied

#### Day 4–5 — Ship N-best rescoring (Tier 1)
This is the main attack on GPTZero. NeurIPS 2025 paper shows 87% detection reduction.
- [ ] Add HuggingFace inference call: `roberta-base-openai-detector`
- [ ] In `humanize()`, generate K=5 candidates per chunk at temps [0.5, 0.6, 0.7, 0.8, 0.9]
- [ ] Score each with RoBERTa, pick the lowest-scored candidate
- [ ] Wire into pipeline in place of single-candidate structural/semantic passes
- [ ] Benchmark: does GPTZero score improve?

#### Day 6 — Non-GPT fingerprint-breaker (if N-best isn't enough)
- [ ] Test DeepSeek V3 or Kimi K2 as the mid-chain paraphraser
      (non-Google, non-GPT lineage → detectors trained on ChatGPT/Gemini have less signal)
- [ ] Add to pipeline only if benchmark improves on all 4 detectors — not just one

#### Day 7 — Consolidate
- [ ] Update benchmark spreadsheet with final scores
- [ ] Update this LAGOM.md with current state
- [ ] Fix GitHub→Vercel auto-deploy
- [ ] Decide: is Tier 3 (fine-tuning) needed?

### Week 2+ (Tier 3 — only if Week 1 hits ceiling)
- Fine-tune Qwen2.5-7B on (AI→human) pairs via Unsloth/LoRA on RunPod
- Dataset: 500+ essay pairs, scored with RoBERTa proxy
- Deploy fine-tuned model to replace Gemini structural pass
- Cost: ~$20–40 RunPod GPU time

### Not on the roadmap (explicitly excluded)
- More prompt engineering — 35+ commits confirmed diminishing returns
- New deterministic passes — whack-a-mole trap
- UI/UX work — fix scores first
- Deep mode changes — fix fast mode fundamentals first

### Success criteria (end of Week 1)
- **Must-have:** QuillBot <15%, ZeroGPT <15% on all 10 benchmark essays
- **Target:** GPTZero <50% on at least 5 of 10 essays
- **Stretch:** All 4 detectors <25% on 3 of 10 essays

---

## Session Log

> Update this section at the end of every session with: date, what changed, and current scores.

### 2026-07-11 (part 1)
- **Changes:** Grammar bug fixed (openerDiversityPass "The fact" → "Such", commit d93b077). Real UI restored after Prash CI sabotage. NVIDIA gpt-oss-20b pass added (commit c885680) — immediately caused score regression. Created LAGOM.md (commit ed20717) after auditing all 160+ commits — confirmed no rebuild needed, peak pipeline (milestone-41) is already in the code.
- **Scores:** QuillBot 78% (regressed from 6%), ZeroGPT 57.8% (regressed from 8%), GPTZero still bad
- **Root cause identified:** gpt-oss-20b is GPT-lineage; QuillBot/ZeroGPT trained to detect ChatGPT outputs
- **Next action:** Unset NVIDIA_API_KEY on Vercel + run benchmark

### 2026-07-11 (part 2)
- **Changes:** Removed `NVIDIA_API_KEY` from Vercel + redeployed. Retested on real QuillBot/ZeroGPT with 400-word and 900-word essays.
- **Scores after NVIDIA removal:** ZeroGPT 4% / 8.8% (recovered to near-target) — **QuillBot 49% / 80% (still broken, barely moved from the regressed 78%)**
- **Key finding:** NVIDIA was NOT the (sole) cause of the QuillBot regression. ZeroGPT doesn't weight punctuation/grammar as heavily as QuillBot's detector does. Root cause traced to real grammar bugs in the deterministic pass stack, visible directly in QuillBot's highlighted flagged text.
- **4 bugs found and fixed (commit 4a08479), all confirmed via direct before/after testing:**
  1. `structuralPerplexityInjector` Mode C — comma→semicolon swap fired before dependent clauses ("which", "because") and mid-list conjunctions, producing ungrammatical semicolons. Disabled (joins Modes A/B as fully off).
  2. `OPENER_SWAPS` — "the"→"Each"/"That" and "both"→"Each"/"Either" broke singular/plural agreement ("Each relational and emotional aspects..."). Replaced with number-agnostic alternatives.
  3. "However," strip in `patternKillerPass` didn't recapitalize the next word, leaving lowercase paragraph starts.
  4. `burstinessInjector` Strategy A split Oxford-comma list sentences ("X, Y, and Z are proving...") into a bare fragment + orphaned sentence. Now requires the split comma to be the sentence's only comma.
  5. **Bigger architectural bug**: `splitIntoVariableChunks` divides long paragraphs into multiple LLM-processing chunks, but `humanize()` rejoined ALL chunks with `\n\n` regardless of origin — fabricating paragraph breaks mid-paragraph. Fixed by tracking paragraph membership (`isNewParagraph[]`) through the chunk pipeline.
  6. Added a general `capitalizeSentenceStarts()` safety net in `grammarAndDedupPass` — catches lowercase-after-period artifacts from any phrase-replacement table (this bug class showed up 3 separate times from 3 different passes; the safety net covers future occurrences too).
- **Verification:** Ran both test essays locally against the dev server hitting real Gemini before deploying — confirmed clean paragraph structure, no fragments, no lowercase starts, no broken semicolons, in the actual API output (not just code review).
- **Deployed:** commit 4a08479, live on lagom-one.vercel.app
- **Not yet done:** Re-test the same two essays against live QuillBot/ZeroGPT to confirm the fix actually moves the real detector scores (internal heuristic score improved, but that's not the same as confirming on the real detector).
- **Next action:** User to re-run the same 400-word and 900-word essays through QuillBot + ZeroGPT and report scores. If QuillBot drops meaningfully, proceed to Day 1 benchmark harness. If not, the remaining QuillBot signal is likely coming from the LLM output itself (Gemini fingerprint), not the deterministic passes — would point toward Day 6 (non-GPT fingerprint-breaker) sooner than planned.

### [Template for future sessions]
- **Date:** YYYY-MM-DD
- **Changes:** [commits, what was modified]
- **Scores:** QuillBot X%, ZeroGPT X%, GPTZero X%, Originality X%
- **Notes:** [anything surprising, decisions made]
- **Next action:** [what to do next session]
