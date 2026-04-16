// ─── Multi-Detector Scraper ──────────────────────────────────────────────────
// Opens ONE browser, runs all 4 detectors in parallel tabs.
// Only imported by deep-humanizer.ts (Fly.io), never by Vercel routes.

import { chromium, type Browser, type Page } from "playwright";

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

const TIMEOUT = 35_000;
const BROWSER_ARGS = ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"];

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function dismissCookies(page: Page): Promise<void> {
  const btn = page.locator(
    "button:has-text('Accept'), button:has-text('Got it'), button:has-text('OK'), button:has-text('I agree'), button:has-text('Agree'), [class*='cookie'] button"
  ).first();
  if (await btn.isVisible({ timeout: 2500 }).catch(() => false)) {
    await btn.click().catch(() => null);
    await page.waitForTimeout(500);
  }
}

function extractPercent(text: string): number {
  const match = text.match(/(\d{1,3})(?:\.\d+)?\s*%/);
  return match ? Math.round(parseFloat(match[1])) : -1;
}

function fail(error: string): DetectorScore {
  return { score: -1, label: "error", error };
}

// ─── GPTZero ─────────────────────────────────────────────────────────────────

async function scrapeGPTZeroPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    await page.goto("https://gptzero.me", { waitUntil: "domcontentloaded", timeout: TIMEOUT });
    await dismissCookies(page);

    const textarea = page.locator("textarea, [contenteditable='true'], [role='textbox']").first();
    await textarea.waitFor({ timeout: TIMEOUT });
    await textarea.click();
    await textarea.fill(text);

    const checkBtn = page.locator(
      "button:has-text('Check Origin'), button:has-text('Check'), button:has-text('Analyze'), button[type='submit']"
    ).first();
    await checkBtn.waitFor({ timeout: TIMEOUT });
    await checkBtn.click();

    // Wait for results
    await page.waitForTimeout(3000);
    await page.waitForSelector("text=/\\d+%/", { timeout: TIMEOUT }).catch(() => null);
    await page.waitForTimeout(1500);

    const score = await page.evaluate(() => {
      const els = Array.from(document.querySelectorAll("*")).filter(el => {
        const t = el.textContent?.trim() ?? "";
        return /^\d{1,3}%$/.test(t) || /(\d{1,3})%\s*(AI|human)/i.test(t);
      });
      for (const el of els) {
        const m = (el.textContent ?? "").match(/(\d{1,3})%/);
        if (m) return parseInt(m[1], 10);
      }
      const scoreEl = document.querySelector("[data-score], [data-ai-score]");
      if (scoreEl) {
        const val = scoreEl.getAttribute("data-score") ?? scoreEl.getAttribute("data-ai-score");
        if (val) return Math.round(parseFloat(val) * (parseFloat(val) <= 1 ? 100 : 1));
      }
      return -1;
    });

    const label = await page.evaluate(() => {
      const labels = ["Human", "AI", "Mixed", "Likely AI", "Likely Human"];
      for (const l of labels) {
        const el = Array.from(document.querySelectorAll("*")).find(
          e => e.textContent?.includes(l) && (e.children.length === 0 || e.children.length <= 2)
        );
        if (el) return el.textContent?.trim() ?? "";
      }
      return "unknown";
    });

    return score === -1 ? fail("Could not parse GPTZero score") : { score, label };
  } catch (err) {
    return fail(`GPTZero: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── ZeroGPT ─────────────────────────────────────────────────────────────────

async function scrapeZeroGPTPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    await page.goto("https://www.zerogpt.com", { waitUntil: "domcontentloaded", timeout: TIMEOUT });
    await dismissCookies(page);

    // ZeroGPT has a textarea with id or class
    const textarea = page.locator("textarea, #textArea, .text-area-input, [placeholder*='text']").first();
    await textarea.waitFor({ timeout: TIMEOUT });
    await textarea.click();
    await textarea.fill(text);

    // Click detect button
    const detectBtn = page.locator(
      "button:has-text('Detect Text'), button:has-text('Detect'), button:has-text('Check'), button[type='submit']"
    ).first();
    await detectBtn.waitFor({ timeout: TIMEOUT });
    await detectBtn.click();

    // Wait for results
    await page.waitForTimeout(3000);
    await page.waitForSelector("text=/\\d+.*%/", { timeout: TIMEOUT }).catch(() => null);
    await page.waitForTimeout(1500);

    const score = await page.evaluate(() => {
      // ZeroGPT shows "X% AI" or "Your text is X% AI GPT generated"
      const resultEls = Array.from(document.querySelectorAll("*")).filter(el => {
        const t = el.textContent?.trim() ?? "";
        return /(\d{1,3}(?:\.\d+)?)\s*%/.test(t) && /AI|GPT|generated|detected/i.test(t);
      });
      for (const el of resultEls) {
        const m = (el.textContent ?? "").match(/(\d{1,3}(?:\.\d+)?)\s*%/);
        if (m) return Math.round(parseFloat(m[1]));
      }
      // Fallback: any percentage visible
      const allPercent = Array.from(document.querySelectorAll("*")).filter(el => {
        const t = el.textContent?.trim() ?? "";
        return /^\d{1,3}(\.\d+)?%$/.test(t);
      });
      for (const el of allPercent) {
        const m = (el.textContent ?? "").match(/(\d{1,3}(?:\.\d+)?)/);
        if (m) return Math.round(parseFloat(m[1]));
      }
      return -1;
    });

    return score === -1 ? fail("Could not parse ZeroGPT score") : { score, label: `${score}% AI` };
  } catch (err) {
    return fail(`ZeroGPT: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── QuillBot ────────────────────────────────────────────────────────────────

async function scrapeQuillBotPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    await page.goto("https://quillbot.com/ai-content-detector", {
      waitUntil: "domcontentloaded",
      timeout: TIMEOUT,
    });
    await dismissCookies(page);

    // QuillBot AI detector has a textarea/input area
    const textarea = page.locator(
      "textarea, [contenteditable='true'], [role='textbox'], .ql-editor, #inputText"
    ).first();
    await textarea.waitFor({ timeout: TIMEOUT });
    await textarea.click();
    await textarea.fill(text);

    // Click scan/check button
    const scanBtn = page.locator(
      "button:has-text('Scan'), button:has-text('Check'), button:has-text('Analyze'), button:has-text('Detect'), button[type='submit']"
    ).first();
    await scanBtn.waitFor({ timeout: TIMEOUT });
    await scanBtn.click();

    // Wait for results
    await page.waitForTimeout(4000);
    await page.waitForSelector("text=/\\d+%/", { timeout: TIMEOUT }).catch(() => null);
    await page.waitForTimeout(1500);

    const score = await page.evaluate(() => {
      // QuillBot shows "X% Human Generated" or "X% AI Generated"
      const allEls = Array.from(document.querySelectorAll("*"));

      // Look for AI percentage specifically
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const aiMatch = t.match(/(\d{1,3})%\s*AI/i);
        if (aiMatch) return parseInt(aiMatch[1], 10);
      }

      // Look for Human percentage and invert
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const humanMatch = t.match(/(\d{1,3})%\s*Human/i);
        if (humanMatch) return 100 - parseInt(humanMatch[1], 10);
      }

      // Fallback: look for any percentage near result indicators
      const resultArea = document.querySelector("[class*='result'], [class*='score'], [class*='detection']");
      if (resultArea) {
        const m = resultArea.textContent?.match(/(\d{1,3})%/);
        if (m) return parseInt(m[1], 10);
      }

      return -1;
    });

    return score === -1 ? fail("Could not parse QuillBot score") : { score, label: `${score}% AI` };
  } catch (err) {
    return fail(`QuillBot: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── Originality.ai ──────────────────────────────────────────────────────────

async function scrapeOriginalityPage(page: Page, text: string): Promise<DetectorScore> {
  try {
    await page.goto("https://originality.ai/ai-checker", {
      waitUntil: "domcontentloaded",
      timeout: TIMEOUT,
    });
    await dismissCookies(page);

    const textarea = page.locator(
      "textarea, [contenteditable='true'], [role='textbox'], #content-input"
    ).first();
    await textarea.waitFor({ timeout: TIMEOUT });
    await textarea.click();
    await textarea.fill(text);

    // Click scan button
    const scanBtn = page.locator(
      "button:has-text('Scan'), button:has-text('Check'), button:has-text('Analyze'), button:has-text('Detect'), button[type='submit']"
    ).first();
    await scanBtn.waitFor({ timeout: TIMEOUT });
    await scanBtn.click();

    // Wait for results
    await page.waitForTimeout(5000);
    await page.waitForSelector("text=/\\d+%/", { timeout: TIMEOUT }).catch(() => null);
    await page.waitForTimeout(2000);

    const score = await page.evaluate(() => {
      const allEls = Array.from(document.querySelectorAll("*"));

      // Originality shows "X% AI" or "Original Score: X%"
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const aiMatch = t.match(/(\d{1,3})%\s*AI/i);
        if (aiMatch) return parseInt(aiMatch[1], 10);
      }

      // Look for "Original" percentage and invert
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        const origMatch = t.match(/(\d{1,3})%\s*Original/i);
        if (origMatch) return 100 - parseInt(origMatch[1], 10);
      }

      // Fallback
      for (const el of allEls) {
        const t = el.textContent?.trim() ?? "";
        if (/^\d{1,3}%$/.test(t)) return parseInt(t, 10);
      }

      return -1;
    });

    return score === -1 ? fail("Could not parse Originality score") : { score, label: `${score}% AI` };
  } catch (err) {
    return fail(`Originality: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ─── Coordinator ─────────────────────────────────────────────────────────────
// Opens one browser, runs all 4 scrapers in parallel tabs, returns all scores.

export async function scoreAllDetectors(text: string): Promise<AllScores> {
  const browser = await chromium.launch({
    headless: true,
    args: BROWSER_ARGS,
  });

  try {
    const [p1, p2, p3, p4] = await Promise.all([
      browser.newPage(),
      browser.newPage(),
      browser.newPage(),
      browser.newPage(),
    ]);

    // Set reasonable viewport for all pages
    await Promise.all([p1, p2, p3, p4].map(p => p.setViewportSize({ width: 1280, height: 800 })));

    const [gptzero, zerogpt, quillbot, originality] = await Promise.all([
      scrapeGPTZeroPage(p1, text),
      scrapeZeroGPTPage(p2, text),
      scrapeQuillBotPage(p3, text),
      scrapeOriginalityPage(p4, text),
    ]);

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
    const page = await browser.newPage();
    await page.setViewportSize({ width: 1280, height: 800 });
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
