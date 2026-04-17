// ─── Multi-Detector Scraper ──────────────────────────────────────────────────
// Opens ONE browser, runs all 4 detectors in parallel tabs (one context each).
// Each context gets a realistic user-agent to avoid bot detection.

import { chromium, type Browser, type BrowserContext, type Page } from "playwright";

export interface DetectorScore {
  score: number;   // 0–100 (% AI), -1 if scraper failed
  label: string;
  error?: string;
}

export interface AllScores {
  gptzero: DetectorScore;
  zerogpt: DetectorScore;
  quillbot: DetectorScore;
  originality: DetectorScore;
}

const TIMEOUT = 40_000;
const WAIT_FOR_RESULT = 8_000;   // ms to wait after clicking submit

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
  // Remove webdriver flag
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
}

// Try fill() first, fall back to click+selectAll+type for React/custom inputs
async function fillInput(page: Page, locator: import("playwright").Locator, text: string): Promise<void> {
  try {
    await locator.click({ timeout: 8000 });
    await locator.fill(text, { timeout: 8000 });
  } catch {
    // fallback: type character-by-character (slower but works with more sites)
    try {
      await locator.click({ timeout: 5000 });
      await page.keyboard.type(text.slice(0, 800), { delay: 0 }); // type first 800 chars max
    } catch { /* ignore */ }
  }
}

function fail(error: string): DetectorScore {
  return { score: -1, label: "error", error };
}

// Extract first percentage from a string
function extractPercent(text: string): number {
  const match = text.match(/(\d{1,3})(?:\.\d+)?\s*%/);
  return match ? Math.round(parseFloat(match[1])) : -1;
}

// ─── GPTZero ─────────────────────────────────────────────────────────────────

async function scrapeGPTZeroPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    console.log("[gptzero] navigating...");
    await page.goto("https://gptzero.me", { waitUntil: "domcontentloaded", timeout: TIMEOUT });
    console.log("[gptzero] title:", await page.title());
    await dismissCookies(page);

    // GPTZero has a large textarea on the main page
    const textarea = page.locator(
      "textarea, [contenteditable='true'], [role='textbox'], [data-testid='text-input']"
    ).first();

    const found = await textarea.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[gptzero] textarea found:", found);
    if (!found) return fail("GPTZero: textarea not found");

    await fillInput(page, textarea, text);
    await page.waitForTimeout(500);

    // Find and click the check button
    const checkBtn = page.locator(
      "button:has-text('Check Origin'), button:has-text('Check for AI'), button:has-text('Check'), button:has-text('Analyze'), button[type='submit']"
    ).first();

    const btnFound = await checkBtn.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[gptzero] checkBtn found:", btnFound);
    if (!btnFound) return fail("GPTZero: submit button not found");

    await checkBtn.click();
    console.log("[gptzero] clicked submit, waiting for result...");
    await page.waitForTimeout(WAIT_FOR_RESULT);

    // Parse score from page
    const score = await page.evaluate(() => {
      const allEls = Array.from(document.querySelectorAll("*"));

      // Look for elements showing "X% AI" or just "X%"
      for (const el of allEls) {
        if (el.children.length > 3) continue; // skip containers
        const t = el.textContent?.trim() ?? "";
        if (/^\d{1,3}%$/.test(t)) {
          const n = parseInt(t, 10);
          if (n >= 0 && n <= 100) return n;
        }
        const aiMatch = t.match(/(\d{1,3})%\s*(?:AI|ai)/);
        if (aiMatch) return parseInt(aiMatch[1], 10);
      }

      // data attribute fallback
      for (const el of allEls) {
        const v = el.getAttribute("data-score") ?? el.getAttribute("data-ai-score") ?? el.getAttribute("data-ai");
        if (v !== null) {
          const n = parseFloat(v);
          if (!isNaN(n)) return Math.round(n <= 1 ? n * 100 : n);
        }
      }

      return -1;
    });

    console.log("[gptzero] score:", score);
    if (score === -1) {
      // Log page content snippet for debugging
      const snippet = await page.evaluate(() => document.body.innerText.slice(0, 300));
      console.log("[gptzero] page snippet:", snippet);
    }

    return score === -1 ? fail("GPTZero: could not parse score") : { score, label: `${score}% AI` };
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

    await detectBtn.click();
    console.log("[zerogpt] clicked submit, waiting...");
    await page.waitForTimeout(WAIT_FOR_RESULT);

    const score = await page.evaluate(() => {
      const allEls = Array.from(document.querySelectorAll("*"));

      // ZeroGPT shows "X% AI GPT" or "Your text is X% AI"
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const m = t.match(/(\d{1,3}(?:\.\d+)?)\s*%\s*(?:AI|GPT|generated)/i);
        if (m) return Math.round(parseFloat(m[1]));
      }

      // Fallback: any standalone percentage in result area
      const resultArea = document.querySelector("[class*='result' i], [class*='score' i], [id*='result' i]");
      if (resultArea) {
        const m = resultArea.textContent?.match(/(\d{1,3}(?:\.\d+)?)\s*%/);
        if (m) return Math.round(parseFloat(m[1]));
      }

      // Last resort: any clean percentage element
      for (const el of allEls) {
        if (el.children.length > 2) continue;
        const t = el.textContent?.trim() ?? "";
        if (/^\d{1,3}(\.\d+)?%$/.test(t)) return Math.round(parseFloat(t));
      }
      return -1;
    });

    console.log("[zerogpt] score:", score);
    if (score === -1) {
      const snippet = await page.evaluate(() => document.body.innerText.slice(0, 300));
      console.log("[zerogpt] page snippet:", snippet);
    }

    return score === -1 ? fail("ZeroGPT: could not parse score") : { score, label: `${score}% AI` };
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

    const textarea = page.locator(
      ".ql-editor, [contenteditable='true'], textarea, [role='textbox']"
    ).first();

    const found = await textarea.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[quillbot] textarea found:", found);
    if (!found) return fail("QuillBot: input area not found");

    await fillInput(page, textarea, text);
    await page.waitForTimeout(500);

    const scanBtn = page.locator(
      "button:has-text('Scan'), button:has-text('Check'), button:has-text('Analyze'), button:has-text('Detect')"
    ).first();

    const btnFound = await scanBtn.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[quillbot] scanBtn found:", btnFound);
    if (!btnFound) return fail("QuillBot: scan button not found");

    await scanBtn.click();
    console.log("[quillbot] clicked submit, waiting...");
    await page.waitForTimeout(WAIT_FOR_RESULT + 2000); // QuillBot is slower

    const score = await page.evaluate(() => {
      const allEls = Array.from(document.querySelectorAll("*"));

      // QuillBot shows "X% AI Content" or "X% Human Content"
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

      // Result container fallback
      const resultArea = document.querySelector("[class*='result' i], [class*='score' i], [class*='detection' i]");
      if (resultArea) {
        const m = resultArea.textContent?.match(/(\d{1,3})%/);
        if (m) return parseInt(m[1], 10);
      }

      return -1;
    });

    console.log("[quillbot] score:", score);
    if (score === -1) {
      const snippet = await page.evaluate(() => document.body.innerText.slice(0, 300));
      console.log("[quillbot] page snippet:", snippet);
    }

    return score === -1 ? fail("QuillBot: could not parse score") : { score, label: `${score}% AI` };
  } catch (err) {
    console.log("[quillbot] error:", err instanceof Error ? err.message : String(err));
    return fail(`QuillBot: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── Originality.ai ──────────────────────────────────────────────────────────

async function scrapeOriginalityPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    console.log("[originality] navigating...");
    await page.goto("https://originality.ai/ai-checker", {
      waitUntil: "domcontentloaded",
      timeout: TIMEOUT,
    });
    console.log("[originality] title:", await page.title());
    await dismissCookies(page);

    const textarea = page.locator(
      "textarea, [contenteditable='true'], [role='textbox'], #content-input, [placeholder*='text' i], [placeholder*='paste' i]"
    ).first();

    const found = await textarea.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[originality] textarea found:", found);
    if (!found) return fail("Originality: textarea not found");

    await fillInput(page, textarea, text);
    await page.waitForTimeout(500);

    const scanBtn = page.locator(
      "button:has-text('Scan'), button:has-text('Check'), button:has-text('Analyze'), button:has-text('Detect'), button[type='submit']"
    ).first();

    const btnFound = await scanBtn.isVisible({ timeout: 8000 }).catch(() => false);
    console.log("[originality] scanBtn found:", btnFound);
    if (!btnFound) return fail("Originality: scan button not found");

    await scanBtn.click();
    console.log("[originality] clicked submit, waiting...");
    await page.waitForTimeout(WAIT_FOR_RESULT + 2000);

    const score = await page.evaluate(() => {
      const allEls = Array.from(document.querySelectorAll("*"));

      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const aiMatch = t.match(/(\d{1,3})%\s*(?:AI|artificial)/i);
        if (aiMatch) return parseInt(aiMatch[1], 10);
      }
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const origMatch = t.match(/(\d{1,3})%\s*(?:original|human)/i);
        if (origMatch) return 100 - parseInt(origMatch[1], 10);
      }

      for (const el of allEls) {
        if (el.children.length > 2) continue;
        const t = el.textContent?.trim() ?? "";
        if (/^\d{1,3}%$/.test(t)) return parseInt(t, 10);
      }
      return -1;
    });

    console.log("[originality] score:", score);
    if (score === -1) {
      const snippet = await page.evaluate(() => document.body.innerText.slice(0, 300));
      console.log("[originality] page snippet:", snippet);
    }

    return score === -1 ? fail("Originality: could not parse score") : { score, label: `${score}% AI` };
  } catch (err) {
    console.log("[originality] error:", err instanceof Error ? err.message : String(err));
    return fail(`Originality: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── Coordinator ─────────────────────────────────────────────────────────────
// Opens one browser, gives each scraper its own stealth context (separate cookies/fingerprint).

export async function scoreAllDetectors(text: string): Promise<AllScores> {
  const browser = await chromium.launch({
    headless: true,
    args: BROWSER_ARGS,
  });

  try {
    // Each detector gets its own context to avoid shared state / bot fingerprinting
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

    const [gptzero, zerogpt, quillbot, originality] = await Promise.all([
      scrapeGPTZeroPage(p1, text),
      scrapeZeroGPTPage(p2, text),
      scrapeQuillBotPage(p3, text),
      scrapeOriginalityPage(p4, text),
    ]);

    console.log(`[scrapers] gptzero=${gptzero.score} zerogpt=${zerogpt.score} quillbot=${quillbot.score} originality=${originality.score}`);
    if (gptzero.error) console.log(`[scrapers] gptzero-err: ${gptzero.error}`);
    if (zerogpt.error) console.log(`[scrapers] zerogpt-err: ${zerogpt.error}`);
    if (quillbot.error) console.log(`[scrapers] quillbot-err: ${quillbot.error}`);
    if (originality.error) console.log(`[scrapers] originality-err: ${originality.error}`);

    return { gptzero, zerogpt, quillbot, originality };
  } finally {
    await browser.close();
  }
}

// ─── Single detector scrape (for targeted re-check) ─────────────────────────

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
