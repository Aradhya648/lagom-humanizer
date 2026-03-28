export interface DetectionResult {
  score: number;
  label: "Likely Human" | "Mixed" | "Likely AI" | "Almost Certainly AI";
}

// ─── Signal Lists ────────────────────────────────────────────────────────────

const AI_FILLER_PHRASES = [
  "it is important to note",
  "it is worth noting",
  "in conclusion",
  "furthermore",
  "it is worth mentioning",
  "as an AI language model",
  "delve into",
  "in today's world",
  "it is crucial",
  "in summary",
  "this essay will",
  "it should be noted",
  "needless to say",
  "it goes without saying",
  "in the realm of",
  "in the field of",
  "plays a crucial role",
  "plays a vital role",
  "it is essential",
  "at the end of the day",
  "when it comes to",
  "in terms of",
  "as previously mentioned",
  "as mentioned above",
  "it is worth noting that",
  "one must consider",
  "it can be argued",
  "it is undeniable",
  "it is clear that",
  "it is evident that",
  "in light of",
  "taking into account",
  "it is imperative",
  "in order to",
  "with regards to",
  "in this day and age",
];

// AI-overused transition words (distinct from filler phrases — these are
// single-word formal connectors that AI sprinkles in at high density).
const TRANSITION_WORDS = [
  "however",
  "therefore",
  "moreover",
  "furthermore",
  "additionally",
  "consequently",
  "nevertheless",
  "nonetheless",
  "thus",
  "hence",
  "accordingly",
  "subsequently",
  "conversely",
  "alternatively",
  "notwithstanding",
];

// Sentence-opener words that AI clusters at the start of sentences.
const AI_STARTER_WORDS = [
  "the",
  "this",
  "it",
  "however",
  "moreover",
  "additionally",
  "furthermore",
  "therefore",
  "consequently",
  "in",
];

// ─── Text Helpers ────────────────────────────────────────────────────────────

function getSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10);
}

function getWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 0);
}

// ─── Existing Metrics (5) ────────────────────────────────────────────────────

// 1. Average sentence length — AI writes uniformly at 20-30 words.
function avgSentenceLengthScore(sentences: string[]): number {
  if (sentences.length === 0) return 0;
  const lengths = sentences.map((s) => s.split(/\s+/).length);
  const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  if (avg < 12) return 10;
  if (avg < 18) return 25;
  if (avg < 24) return 50;
  if (avg < 30) return 70;
  return 85;
}

// 2. Burstiness — low std dev = AI (uniform); high std dev = human (varied).
function burstiScore(sentences: string[]): number {
  if (sentences.length < 3) return 50;
  const lengths = sentences.map((s) => s.split(/\s+/).length);
  const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const variance =
    lengths.reduce((sum, l) => sum + Math.pow(l - avg, 2), 0) / lengths.length;
  const stdDev = Math.sqrt(variance);
  if (stdDev > 12) return 5;
  if (stdDev > 8) return 20;
  if (stdDev > 5) return 45;
  if (stdDev > 3) return 65;
  return 85;
}

// 3. Vocabulary diversity (TTR) — low unique:total ratio = AI.
function vocabularyDiversityScore(words: string[]): number {
  if (words.length === 0) return 50;
  const unique = new Set(words).size;
  const ttr = unique / words.length;
  if (ttr > 0.7) return 10;
  if (ttr > 0.6) return 25;
  if (ttr > 0.5) return 45;
  if (ttr > 0.4) return 65;
  return 80;
}

// 4. AI filler phrase frequency — normalised per 100 words.
function fillerPhraseScore(text: string): number {
  const lower = text.toLowerCase();
  const wordCount = text.split(/\s+/).length;
  let hits = 0;
  for (const phrase of AI_FILLER_PHRASES) {
    let idx = 0;
    while ((idx = lower.indexOf(phrase, idx)) !== -1) {
      hits++;
      idx += phrase.length;
    }
  }
  const rate = (hits / Math.max(wordCount, 1)) * 100;
  if (rate === 0) return 10;
  if (rate < 0.5) return 30;
  if (rate < 1) return 55;
  if (rate < 2) return 75;
  return 90;
}

// 5. Sentence starter patterns — AI clusters on a fixed set of openers.
function sentenceStarterScore(sentences: string[]): number {
  if (sentences.length === 0) return 50;
  const starters = sentences.map((s) =>
    s.split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, "")
  );
  const aiStarters = starters.filter((w) => AI_STARTER_WORDS.includes(w));
  const ratio = aiStarters.length / starters.length;
  if (ratio < 0.2) return 10;
  if (ratio < 0.35) return 30;
  if (ratio < 0.5) return 55;
  if (ratio < 0.65) return 72;
  return 88;
}

// ─── New Metrics (4) ─────────────────────────────────────────────────────────

// 6. Punctuation entropy — AI defaults to periods and commas.
//    Humans use semicolons, em-dashes, colons, parentheses with more variety.
//    Shannon entropy over punctuation character distribution.
function punctuationEntropyScore(text: string): number {
  const punctChars = text.match(/[.,;:!?—–\-()\[\]{}"']/g) ?? [];
  if (punctChars.length < 5) return 50; // too few to score reliably

  const counts: Record<string, number> = {};
  for (const p of punctChars) {
    counts[p] = (counts[p] ?? 0) + 1;
  }

  const total = punctChars.length;
  let entropy = 0;
  for (const count of Object.values(counts)) {
    const p = count / total;
    entropy -= p * Math.log2(p);
  }

  // Higher entropy = more varied punctuation = more human
  if (entropy > 2.5) return 10;
  if (entropy > 2.0) return 25;
  if (entropy > 1.5) return 45;
  if (entropy > 1.0) return 65;
  return 80;
}

// 7. Consecutive repeated openers — AI often starts adjacent sentences the
//    same way even after prompt-level variation instructions.
function repeatedOpenersScore(sentences: string[]): number {
  if (sentences.length < 3) return 30;

  let repeatedPairs = 0;
  for (let i = 0; i < sentences.length - 1; i++) {
    const w1 = sentences[i]
      .split(/\s+/)[0]
      .toLowerCase()
      .replace(/[^a-z]/g, "");
    const w2 = sentences[i + 1]
      .split(/\s+/)[0]
      .toLowerCase()
      .replace(/[^a-z]/g, "");
    if (w1.length > 0 && w1 === w2) repeatedPairs++;
  }

  const ratio = repeatedPairs / (sentences.length - 1);
  if (ratio === 0) return 10;
  if (ratio < 0.1) return 25;
  if (ratio < 0.2) return 50;
  if (ratio < 0.35) return 70;
  return 88;
}

// 8. Transition word density — AI sprinkles formal connectors at 2-5× the
//    rate of human writers. Measured as count per 100 words.
function transitionDensityScore(text: string): number {
  const lower = text.toLowerCase();
  const wordCount = Math.max(text.split(/\s+/).length, 1);

  let count = 0;
  for (const t of TRANSITION_WORDS) {
    const regex = new RegExp(`\\b${t}\\b`, "g");
    const matches = lower.match(regex);
    if (matches) count += matches.length;
  }

  const rate = (count / wordCount) * 100;
  // Human baseline ~0-0.8 per 100 words; AI baseline ~2-5 per 100 words
  if (rate === 0) return 10;
  if (rate < 0.5) return 20;
  if (rate < 1.0) return 40;
  if (rate < 2.0) return 65;
  if (rate < 3.0) return 80;
  return 90;
}

// 9. Clause length distribution — split on comma/semicolon boundaries as a
//    rough clause proxy. AI produces uniform clause lengths; humans vary them.
function clauseLengthDistributionScore(text: string): number {
  const clauses = text
    .split(/[,;]/)
    .map((c) => c.trim().split(/\s+/).length)
    .filter((len) => len >= 2);

  if (clauses.length < 4) return 50; // not enough signal

  const avg = clauses.reduce((a, b) => a + b, 0) / clauses.length;
  const variance =
    clauses.reduce((sum, l) => sum + Math.pow(l - avg, 2), 0) / clauses.length;
  const stdDev = Math.sqrt(variance);

  // High std dev = human variety. Low std dev = AI uniformity.
  if (stdDev > 8) return 10;
  if (stdDev > 5) return 25;
  if (stdDev > 3) return 50;
  if (stdDev > 2) return 70;
  return 85;
}

// ─── Label + Export ──────────────────────────────────────────────────────────

function getLabel(score: number): DetectionResult["label"] {
  if (score <= 30) return "Likely Human";
  if (score <= 60) return "Mixed";
  if (score <= 80) return "Likely AI";
  return "Almost Certainly AI";
}

export function detectAI(text: string): DetectionResult {
  if (!text || text.trim().length === 0) {
    return { score: 0, label: "Likely Human" };
  }

  const sentences = getSentences(text);
  const words = getWords(text);

  // ── Original 5 metrics ──────────────────────────────────────────
  const avgLen   = avgSentenceLengthScore(sentences);   // 0.12
  const burst    = burstiScore(sentences);              // 0.25
  const vocabDiv = vocabularyDiversityScore(words);     // 0.08
  const filler   = fillerPhraseScore(text);             // 0.18
  const starters = sentenceStarterScore(sentences);     // 0.10

  // ── New 4 metrics ───────────────────────────────────────────────
  const punctEnt    = punctuationEntropyScore(text);         // 0.03
  const repOpeners  = repeatedOpenersScore(sentences);       // 0.07
  const transDens   = transitionDensityScore(text);          // 0.12
  const clauseDist  = clauseLengthDistributionScore(text);   // 0.05

  // Weights sum to 1.00
  const raw =
    avgLen   * 0.12 +
    burst    * 0.25 +
    vocabDiv * 0.08 +
    filler   * 0.18 +
    starters * 0.10 +
    punctEnt    * 0.03 +
    repOpeners  * 0.07 +
    transDens   * 0.12 +
    clauseDist  * 0.05;

  const score = Math.round(Math.min(100, Math.max(0, raw)));
  return { score, label: getLabel(score) };
}
