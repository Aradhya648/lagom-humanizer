// ─── Multi-Detector Scraper ──────────────────────────────────────────────────
// Opens ONE browser, runs all 4 detectors in parallel tabs (one context each).
// Each context gets a realistic user-agent to avoid bot detection.
// Each scraper returns the overall score AND any sentences the detector
// flagged as AI — used for targeted surgical re-humanization.

import { chromium, type Browser, type BrowserContext, type Page } from "playwright";

export interface DetectorScore {
  score: number;   // 0–100 (% AI), -1 if scraper failed
  label: string;
  error?: string;
  flaggedSentences: string[];  // sentences the detector highlighted as AI
}

export interface AllScores {
  gptzero: DetectorScore;
  zerogpt: DetectorScore;
  quillbot: DetectorScore;
  originality: DetectorScore;
}

const TIMEOUT = 30_000;
const WAIT_FOR_RESULT = 6_000;   // ms to wait after clicking submit

const BROWSER_ARGS = [
  "--no-sandbox",
  "--disable-setuid-sandbox",
  "--disable-dev-shm-usage",
  "--disable-blink-features=AutomationControlled",
  "--disable-features=IsolateOrigins,site-per-process",
];

// Realistic Chrome user-agent to avoid bot detection
const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function newStealthContext(browser: Browser): Promise<BrowserContext> {
  const ctx = await browser.newContext({
    userAgent: USER_AGENT,
    viewport: { width: 1280, height: 800 },
    locale: "en-US",
    timezoneId: "America/New_York",
    extraHTTPHeaders: {
      "Accept-Language": "en-US,en;q=0.9",
    },
  });
  await ctx.addInitScript(() => {
    Object.defineProperty(navigator, "webdriver", { get: () => false });
    Object.defineProperty(navigator, "plugins", { get: () => [1, 2, 3] });
    Object.defineProperty(navigator, "languages", { get: () => ["en-US", "en"] });
  });
  return ctx;
}

async function dismissCookies(page: Page): Promise<void> {
  try {
    const btn = page.locator(
      "button:has-text('Accept'), button:has-text('Accept all'), button:has-text('Got it'), button:has-text('OK'), button:has-text('I agree'), button:has-text('Agree'), button:has-text('Allow'), [class*='cookie'] button, [id*='cookie'] button"
    ).first();
    if (await btn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await btn.click().catch(() => null);
      await page.waitForTimeout(600);
    }
  } catch { /* ignore */ }

  // Remove sticky overlays/footers that intercept pointer events
  await page.evaluate(() => {
    const selectors = [
      "#fs-sticky-footer",
      ".fs-sticky-footer",
      "[id*='cookie']",
      "[class*='cookie-banner']",
      "[class*='consent']",
      "[class*='gdpr']",
      "[id*='gdpr']",
      "[class*='sticky-footer']",
      "[id*='sticky-footer']",
      "[class*='overlay']",
      "[class*='modal-backdrop']",
    ];
    for (const sel of selectors) {
      document.querySelectorAll(sel).forEach(el => (el as HTMLElement).remove());
    }
  }).catch(() => null);
}

async function fillInput(page: Page, locator: import("playwright").Locator, text: string): Promise<void> {
  try {
    await locator.click({ timeout: 8000 });
    await locator.fill(text, { timeout: 8000 });
  } catch {
    try {
      await locator.click({ timeout: 5000 });
      await page.keyboard.type(text.slice(0, 800), { delay: 0 });
    } catch { /* ignore */ }
  }
}

function fail(error: string): DetectorScore {
  return { score: -1, label: "error", error, flaggedSentences: [] };
}

// ─── GPTZero ─────────────────────────────────────────────────────────────────

async function scrapeGPTZeroPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    console.log("[gptzero] navigating...");
    await page.goto("https://gptzero.me", { waitUntil: "domcontentloaded", timeout: TIMEOUT });
    console.log("[gptzero] title:", await page.title());
    await dismissCookies(page);

    const textarea = page.locator(
      "textarea, [contenteditable='true'], [role='textbox'], [data-testid='text-input']"
    ).first();

    const found = await textarea.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[gptzero] textarea found:", found);
    if (!found) return fail("GPTZero: textarea not found");

    await fillInput(page, textarea, text);
    await page.waitForTimeout(500);

    const checkBtn = page.locator(
      "button:has-text('Check Origin'), button:has-text('Check for AI'), button:has-text('Check'), button:has-text('Analyze'), button[type='submit']"
    ).first();

    const btnFound = await checkBtn.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[gptzero] checkBtn found:", btnFound);
    if (!btnFound) return fail("GPTZero: submit button not found");

    await checkBtn.scrollIntoViewIfNeeded().catch(() => null);
    await checkBtn.click({ force: true, timeout: 15000 });
    console.log("[gptzero] clicked submit, waiting for result...");
    await page.waitForTimeout(WAIT_FOR_RESULT);

    const score = await page.evaluate(() => {
      const allEls = Array.from(document.querySelectorAll("*"));

      // Strategy 1: "AI 73%" format — new GPTZero app UI (label comes BEFORE number)
      for (const el of allEls) {
        if (el.children.length > 3) continue;
        const t = el.textContent?.trim() ?? "";
        const labelFirst = t.match(/\bAI\s+(\d{1,3})\s*%/i);
        if (labelFirst) { const n = parseInt(labelFirst[1], 10); if (n >= 0 && n <= 100) return n; }
      }

      // Strategy 2: "73% AI" format — old UI
      for (const el of allEls) {
        if (el.children.length > 3) continue;
        const t = el.textContent?.trim() ?? "";
        const labelAfter = t.match(/(\d{1,3})\s*%\s*(?:AI|ai)\b/);
        if (labelAfter) { const n = parseInt(labelAfter[1], 10); if (n >= 0 && n <= 100) return n; }
      }

      // Strategy 3: bare percentage in a leaf element
      for (const el of allEls) {
        if (el.children.length > 2) continue;
        const t = el.textContent?.trim() ?? "";
        if (/^\d{1,3}%$/.test(t)) { const n = parseInt(t, 10); if (n >= 0 && n <= 100) return n; }
      }

      // Strategy 4: data attributes
      for (const el of allEls) {
        const v = el.getAttribute("data-score") ?? el.getAttribute("data-ai-score") ?? el.getAttribute("data-ai");
        if (v !== null) { const n = parseFloat(v); if (!isNaN(n)) return Math.round(n <= 1 ? n * 100 : n); }
      }

      // Strategy 5: infer from Human% — "Human 27%" → AI = 73%
      for (const el of allEls) {
        if (el.children.length > 3) continue;
        const t = el.textContent?.trim() ?? "";
        const humanFirst = t.match(/\bHuman\s+(\d{1,3})\s*%/i);
        if (humanFirst) { const n = parseInt(humanFirst[1], 10); if (n >= 0 && n <= 100) return 100 - n; }
        const humanAfter = t.match(/(\d{1,3})\s*%\s*Human\b/i);
        if (humanAfter) { const n = parseInt(humanAfter[1], 10); if (n >= 0 && n <= 100) return 100 - n; }
      }

      return -1;
    });

    // Extract flagged sentences — GPTZero highlights AI sentences with specific classes
    const flaggedSentences = await page.evaluate(() => {
      const results: string[] = [];
      // Strategy 1: elements with AI-indicator classes
      const cssSelectors = [
        '[class*="sentence-ai"]',
        '[class*="ai-sentence"]',
        '[class*="highlight"]',
        '[class*="aiSentence"]',
        '[class*="detected"]',
        '[data-ai="true"]',
        '[data-result="ai"]',
        'span[style*="background-color"]',
        'span[style*="background:"]',
        'mark',
      ];
      for (const sel of cssSelectors) {
        const els = Array.from(document.querySelectorAll(sel));
        for (const el of els) {
          const t = el.textContent?.trim() ?? "";
          if (t.length > 20 && t.length < 600) results.push(t);
        }
      }
      // Dedup
      return Array.from(new Set(results)).slice(0, 15);
    });

    console.log("[gptzero] score:", score, "flagged:", flaggedSentences.length);

    return score === -1
      ? fail("GPTZero: could not parse score")
      : { score, label: `${score}% AI`, flaggedSentences };
  } catch (err) {
    console.log("[gptzero] error:", err instanceof Error ? err.message : String(err));
    return fail(`GPTZero: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── ZeroGPT ─────────────────────────────────────────────────────────────────

async function scrapeZeroGPTPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    console.log("[zerogpt] navigating...");
    await page.goto("https://www.zerogpt.com", { waitUntil: "domcontentloaded", timeout: TIMEOUT });
    console.log("[zerogpt] title:", await page.title());
    await dismissCookies(page);

    const textarea = page.locator(
      "textarea, #textArea, [placeholder*='text' i], [placeholder*='paste' i]"
    ).first();

    const found = await textarea.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[zerogpt] textarea found:", found);
    if (!found) return fail("ZeroGPT: textarea not found");

    await fillInput(page, textarea, text);
    await page.waitForTimeout(500);

    const detectBtn = page.locator(
      "button:has-text('Detect Text'), button:has-text('Detect'), button:has-text('Check'), button[type='submit']"
    ).first();

    const btnFound = await detectBtn.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[zerogpt] detectBtn found:", btnFound);
    if (!btnFound) return fail("ZeroGPT: submit button not found");

    await detectBtn.scrollIntoViewIfNeeded().catch(() => null);
    await detectBtn.click({ force: true, timeout: 15000 });
    console.log("[zerogpt] clicked submit, waiting...");
    await page.waitForTimeout(WAIT_FOR_RESULT);

    const score = await page.evaluate(() => {
      const allEls = Array.from(document.querySelectorAll("*"));
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const m = t.match(/(\d{1,3}(?:\.\d+)?)\s*%\s*(?:AI|GPT|generated)/i);
        if (m) return Math.round(parseFloat(m[1]));
      }
      const resultArea = document.querySelector("[class*='result' i], [class*='score' i], [id*='result' i]");
      if (resultArea) {
        const m = resultArea.textContent?.match(/(\d{1,3}(?:\.\d+)?)\s*%/);
        if (m) return Math.round(parseFloat(m[1]));
      }
      for (const el of allEls) {
        if (el.children.length > 2) continue;
        const t = el.textContent?.trim() ?? "";
        if (/^\d{1,3}(\.\d+)?%$/.test(t)) return Math.round(parseFloat(t));
      }
      return -1;
    });

    // ZeroGPT highlights AI sentences in yellow — extract those
    const flaggedSentences = await page.evaluate(() => {
      const results: string[] = [];
      const cssSelectors = [
        'mark',
        '[style*="background-color: yellow"]',
        '[style*="background-color:yellow"]',
        '[style*="background: yellow"]',
        '[style*="background:yellow"]',
        '[style*="rgb(255, 255, 0)"]',
        '[style*="rgb(255,255,0)"]',
        '[style*="#ffff"]',
        '[style*="#FFFF"]',
        '[class*="highlight"]',
        '[class*="flagged"]',
        '[class*="ai-text"]',
      ];
      for (const sel of cssSelectors) {
        const els = Array.from(document.querySelectorAll(sel));
        for (const el of els) {
          const t = el.textContent?.trim() ?? "";
          if (t.length > 20 && t.length < 600) results.push(t);
        }
      }
      return Array.from(new Set(results)).slice(0, 15);
    });

    console.log("[zerogpt] score:", score, "flagged:", flaggedSentences.length);

    return score === -1
      ? fail("ZeroGPT: could not parse score")
      : { score, label: `${score}% AI`, flaggedSentences };
  } catch (err) {
    console.log("[zerogpt] error:", err instanceof Error ? err.message : String(err));
    return fail(`ZeroGPT: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── QuillBot ────────────────────────────────────────────────────────────────

async function scrapeQuillBotPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    console.log("[quillbot] navigating...");
    await page.goto("https://quillbot.com/ai-content-detector", {
      waitUntil: "domcontentloaded",
      timeout: TIMEOUT,
    });
    console.log("[quillbot] title:", await page.title());
    await dismissCookies(page);
    await page.waitForTimeout(2500);  // React hydration

    const textarea = page.locator([
      ".ql-editor",
      "[contenteditable='true']",
      "[data-slate-editor='true']",
      "[data-gramm='false']",
      ".ProseMirror",
      "[aria-multiline='true']",
      "[data-placeholder]",
      "[role='textbox']",
      "textarea",
      "[class*='editor']",
      "[class*='input']",
      "[class*='text-area']",
    ].join(", ")).first();

    const found = await textarea.isVisible({ timeout: 12000 }).catch(() => false);
    console.log("[quillbot] textarea found:", found);

    if (!found) {
      // JS fallback: find any editable element and inject text
      const injected = await page.evaluate((txt) => {
        const el = document.querySelector(
          "[contenteditable='true'], textarea, [role='textbox'], .ql-editor, .ProseMirror"
        ) as HTMLElement | null;
        if (!el) return false;
        el.focus();
        if (el.tagName === "TEXTAREA") (el as HTMLTextAreaElement).value = txt;
        else el.innerText = txt;
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
      }, text);
      if (!injected) return fail("QuillBot: input area not found");
    } else {
      await fillInput(page, textarea, text);
    }
    await page.waitForTimeout(500);

    const scanBtn = page.locator(
      "button:has-text('Scan'), button:has-text('Check'), button:has-text('Analyze'), button:has-text('Detect')"
    ).first();

    const btnFound = await scanBtn.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[quillbot] scanBtn found:", btnFound);
    if (!btnFound) return fail("QuillBot: scan button not found");

    await scanBtn.scrollIntoViewIfNeeded().catch(() => null);
    await scanBtn.click({ force: true, timeout: 15000 });
    console.log("[quillbot] clicked submit, waiting...");
    await page.waitForTimeout(WAIT_FOR_RESULT + 2000); // QuillBot is slower

    const score = await page.evaluate(() => {
      const allEls = Array.from(document.querySelectorAll("*"));
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const aiMatch = t.match(/(\d{1,3})%\s*AI/i);
        if (aiMatch) return parseInt(aiMatch[1], 10);
      }
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const humanMatch = t.match(/(\d{1,3})%\s*Human/i);
        if (humanMatch) return 100 - parseInt(humanMatch[1], 10);
      }
      const resultArea = document.querySelector("[class*='result' i], [class*='score' i], [class*='detection' i]");
      if (resultArea) {
        const m = resultArea.textContent?.match(/(\d{1,3})%/);
        if (m) return parseInt(m[1], 10);
      }
      return -1;
    });

    // QuillBot shows "Main AI Contributors" section with flagged excerpts
    const flaggedSentences = await page.evaluate(() => {
      const results: string[] = [];
      // Look for contributor cards / highlighted sections
      const sectionSelectors = [
        '[class*="contributor" i]',
        '[class*="ai-refined" i]',
        '[class*="ai-generated" i]',
        '[class*="highlight"]',
        '[class*="flagged"]',
        'mark',
        '[style*="background-color"]',
      ];
      for (const sel of sectionSelectors) {
        const els = Array.from(document.querySelectorAll(sel));
        for (const el of els) {
          const t = el.textContent?.trim() ?? "";
          // Skip if it looks like a label (too short) or the full text (too long)
          if (t.length > 30 && t.length < 500 && !/^(AI|Human|Main|Contributors)/i.test(t)) {
            results.push(t);
          }
        }
      }
      return Array.from(new Set(results)).slice(0, 10);
    });

    console.log("[quillbot] score:", score, "flagged:", flaggedSentences.length);

    return score === -1
      ? fail("QuillBot: could not parse score")
      : { score, label: `${score}% AI`, flaggedSentences };
  } catch (err) {
    console.log("[quillbot] error:", err instanceof Error ? err.message : String(err));
    return fail(`QuillBot: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── Originality → Writer.com detector (free, no account) ────────────────────

async function scrapeOriginalityPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    console.log("[originality] navigating to writer.com...");
    await page.goto("https://writer.com/ai-content-detector/", {
      waitUntil: "domcontentloaded",
      timeout: TIMEOUT,
    });
    console.log("[originality] title:", await page.title());
    await dismissCookies(page);
    await page.waitForTimeout(1500);

    const textarea = page.locator(
      "textarea, [contenteditable='true'], [role='textbox'], [placeholder*='text' i], [placeholder*='paste' i], [placeholder*='enter' i], [class*='editor'], [class*='input'], .ProseMirror"
    ).first();

    const found = await textarea.isVisible({ timeout: 10000 }).catch(() => false);
    console.log("[originality/writer] textarea found:", found);

    if (!found) {
      // JS fallback
      const injected = await page.evaluate((txt) => {
        const el = document.querySelector(
          "textarea, [contenteditable='true'], [role='textbox'], .ProseMirror"
        ) as HTMLElement | null;
        if (!el) return false;
        el.focus();
        if (el.tagName === "TEXTAREA") (el as HTMLTextAreaElement).value = txt;
        else el.innerText = txt;
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
      }, text);
      if (!injected) return fail("Writer: textarea not found");
    } else {
      await fillInput(page, textarea, text);
    }
    await page.waitForTimeout(500);

    const scanBtn = page.locator(
      "button:has-text('Analyze'), button:has-text('Check'), button:has-text('Detect'), button:has-text('Scan'), button[type='submit']"
    ).first();

    const btnFound = await scanBtn.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[originality/writer] scanBtn found:", btnFound);
    if (!btnFound) return fail("Writer: scan button not found");

    await scanBtn.scrollIntoViewIfNeeded().catch(() => null);
    await scanBtn.click({ force: true, timeout: 15000 });
    console.log("[originality/writer] clicked submit, waiting...");
    // Writer.com can take 8-12s to process; poll for result instead of a fixed wait
    await page.waitForTimeout(3000);
    let score = -1;
    for (let attempt = 0; attempt < 6 && score === -1; attempt++) {
      await page.waitForTimeout(2000);
      score = await page.evaluate(() => {
        const allEls = Array.from(document.querySelectorAll("*"));

        // "% AI-generated" or "% AI" patterns
        for (const el of allEls) {
          if (el.children.length > 4) continue;
          const t = el.textContent?.trim() ?? "";
          const m = t.match(/(\d{1,3})\s*%\s*(?:AI[- ]?generated|AI\b)/i);
          if (m) { const n = parseInt(m[1], 10); if (n >= 0 && n <= 100) return n; }
        }
        // "% human" / "% original" → invert
        for (const el of allEls) {
          if (el.children.length > 4) continue;
          const t = el.textContent?.trim() ?? "";
          const m = t.match(/(\d{1,3})\s*%\s*(?:human|original)/i);
          if (m) { const n = parseInt(m[1], 10); if (n >= 0 && n <= 100) return 100 - n; }
        }
        // Bare percentage in leaf nodes
        for (const el of allEls) {
          if (el.children.length > 1) continue;
          const t = el.textContent?.trim() ?? "";
          if (/^\d{1,3}%$/.test(t)) { const n = parseInt(t, 10); if (n >= 0 && n <= 100) return n; }
        }
        return -1;
      });
      if (score !== -1) break;
      console.log(`[originality/writer] attempt ${attempt + 1}: score still -1, retrying...`);
    }

    // Writer.com highlights AI sentences in red/orange
    const flaggedSentences = await page.evaluate(() => {
      const results: string[] = [];
      const cssSelectors = [
        'mark',
        '[style*="background-color: red"]',
        '[style*="background-color:red"]',
        '[style*="background: rgb(255"]',
        '[style*="color: rgb(255"]',
        '[class*="highlight"]',
        '[class*="ai-text"]',
        '[class*="flagged"]',
        '[class*="detected"]',
      ];
      for (const sel of cssSelectors) {
        const els = Array.from(document.querySelectorAll(sel));
        for (const el of els) {
          const t = el.textContent?.trim() ?? "";
          if (t.length > 20 && t.length < 600) results.push(t);
        }
      }
      return Array.from(new Set(results)).slice(0, 15);
    });

    console.log("[originality/writer] score:", score, "flagged:", flaggedSentences.length);

    return score === -1
      ? fail("Writer: could not parse score")
      : { score, label: `${score}% AI`, flaggedSentences };
  } catch (err) {
    console.log("[originality] error:", err instanceof Error ? err.message : String(err));
    return fail(`Writer: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── Coordinator ─────────────────────────────────────────────────────────────

// Per-scraper hard ceiling — no individual detector can block the round longer than this
const SCRAPER_HARD_TIMEOUT_MS = 60_000;
// Whole-round hard ceiling — even if all 4 hang, this guarantees the round ends
const SCORE_ALL_HARD_TIMEOUT_MS = 90_000;

function withScraperTimeout(
  name: string,
  p: Promise<DetectorScore>,
): Promise<DetectorScore> {
  return new Promise<DetectorScore>((resolve) => {
    let done = false;
    const timer = setTimeout(() => {
      if (done) return;
      done = true;
      console.log(`[scrapers] ${name} hard-timeout after ${SCRAPER_HARD_TIMEOUT_MS}ms`);
      resolve({ score: -1, label: `${name}: hard timeout`, flaggedSentences: [] });
    }, SCRAPER_HARD_TIMEOUT_MS);
    p.then((v) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve(v);
    }).catch((err) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      console.log(`[scrapers] ${name} rejected:`, err instanceof Error ? err.message : String(err));
      resolve({ score: -1, label: `${name}: error`, flaggedSentences: [] });
    });
  });
}

export async function scoreAllDetectors(text: string): Promise<AllScores> {
  const failAll = (label: string): AllScores => ({
    gptzero: { score: -1, label, flaggedSentences: [] },
    zerogpt: { score: -1, label, flaggedSentences: [] },
    quillbot: { score: -1, label, flaggedSentences: [] },
    originality: { score: -1, label, flaggedSentences: [] },
  });

  const work = (async (): Promise<AllScores> => {
    const browser = await chromium.launch({
      headless: true,
      args: BROWSER_ARGS,
    });

    try {
      const [ctx1, ctx2, ctx3, ctx4] = await Promise.all([
        newStealthContext(browser),
        newStealthContext(browser),
        newStealthContext(browser),
        newStealthContext(browser),
      ]);

      const [p1, p2, p3, p4] = await Promise.all([
        ctx1.newPage(),
        ctx2.newPage(),
        ctx3.newPage(),
        ctx4.newPage(),
      ]);

      // Each scraper is guarded with its own hard timeout so one hang can't stall the round.
      const [gptzero, zerogpt, quillbot, originality] = await Promise.all([
        withScraperTimeout("gptzero", scrapeGPTZeroPage(p1, text)),
        withScraperTimeout("zerogpt", scrapeZeroGPTPage(p2, text)),
        withScraperTimeout("quillbot", scrapeQuillBotPage(p3, text)),
        withScraperTimeout("originality", scrapeOriginalityPage(p4, text)),
      ]);

      console.log(`[scrapers] gptzero=${gptzero.score} zerogpt=${zerogpt.score} quillbot=${quillbot.score} originality=${originality.score}`);
      console.log(`[scrapers] flagged: gptzero=${gptzero.flaggedSentences.length} zerogpt=${zerogpt.flaggedSentences.length} quillbot=${quillbot.flaggedSentences.length} originality=${originality.flaggedSentences.length}`);

      return { gptzero, zerogpt, quillbot, originality };
    } finally {
      await browser.close().catch(() => null);
    }
  })();

  // Outer safety net — guarantees this function resolves no matter what Playwright does.
  return await new Promise<AllScores>((resolve) => {
    let done = false;
    const timer = setTimeout(() => {
      if (done) return;
      done = true;
      console.log(`[scrapers] scoreAllDetectors HARD TIMEOUT after ${SCORE_ALL_HARD_TIMEOUT_MS}ms — returning empty scores`);
      resolve(failAll("round timed out"));
    }, SCORE_ALL_HARD_TIMEOUT_MS);
    work.then((v) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve(v);
    }).catch((err) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      console.log(`[scrapers] scoreAllDetectors rejected:`, err instanceof Error ? err.message : String(err));
      resolve(failAll("round error"));
    });
  });
}

export async function scrapeSingleDetector(
  text: string,
  detector: keyof AllScores
): Promise<DetectorScore> {
  const browser = await chromium.launch({ headless: true, args: BROWSER_ARGS });
  try {
    const ctx = await newStealthContext(browser);
    const page = await ctx.newPage();
    const scrapers = {
      gptzero: scrapeGPTZeroPage,
      zerogpt: scrapeZeroGPTPage,
      quillbot: scrapeQuillBotPage,
      originality: scrapeOriginalityPage,
    };
    return await scrapers[detector](page, text);
  } finally {
    await browser.close();
  }
}
