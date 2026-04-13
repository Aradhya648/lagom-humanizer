import { chromium } from "playwright";

export interface GPTZeroResult {
  score: number;          // 0–100 (% AI detected)
  label: string;         // raw label from site
  error?: string;
}

const GPTZERO_URL = "https://gptzero.me";
const TIMEOUT = 30_000;

/**
 * Scrapes GPTZero detection score for a given text.
 * Requires Playwright + Chromium installed (works on Render, not Vercel).
 */
export async function scrapeGPTZero(text: string): Promise<GPTZeroResult> {
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });

  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: 1280, height: 800 });

    await page.goto(GPTZERO_URL, { waitUntil: "domcontentloaded", timeout: TIMEOUT });

    // Dismiss cookie banner if present
    const cookieBtn = page.locator("button:has-text('Accept'), button:has-text('Got it'), button:has-text('OK')").first();
    if (await cookieBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await cookieBtn.click().catch(() => null);
    }

    // Find the text area and fill it
    const textarea = page.locator("textarea, [contenteditable='true'], [role='textbox']").first();
    await textarea.waitFor({ timeout: TIMEOUT });
    await textarea.click();
    await textarea.fill(text);

    // Click the check button
    const checkBtn = page.locator(
      "button:has-text('Check Origin'), button:has-text('Check'), button:has-text('Analyze'), button[type='submit']"
    ).first();
    await checkBtn.waitFor({ timeout: TIMEOUT });
    await checkBtn.click();

    // Wait for results — look for a percentage element
    await page.waitForSelector(
      "[data-testid='score'], .score, .result-score, .prediction-score, text=/\\d+%/",
      { timeout: TIMEOUT }
    );

    // Small buffer for animations
    await page.waitForTimeout(1500);

    // Extract score — try multiple known selector patterns
    const score = await page.evaluate(() => {
      // Pattern 1: elements with "%" in text near score labels
      const percentEls = Array.from(document.querySelectorAll("*")).filter(el => {
        const text = el.textContent?.trim() ?? "";
        return /^\d{1,3}%$/.test(text) || /^(\d{1,3})% AI/.test(text);
      });

      for (const el of percentEls) {
        const match = (el.textContent ?? "").match(/(\d{1,3})%/);
        if (match) return parseInt(match[1], 10);
      }

      // Pattern 2: data attributes
      const scoreEl = document.querySelector("[data-score], [data-ai-score]");
      if (scoreEl) {
        const val = scoreEl.getAttribute("data-score") ?? scoreEl.getAttribute("data-ai-score");
        if (val) return Math.round(parseFloat(val) * (parseFloat(val) <= 1 ? 100 : 1));
      }

      return -1;
    });

    // Also grab label text
    const label = await page.evaluate(() => {
      const labels = ["Human", "AI", "Mixed", "Likely AI", "Likely Human", "Your text is"];
      for (const l of labels) {
        const el = Array.from(document.querySelectorAll("*")).find(
          e => e.textContent?.includes(l) && (e.children.length === 0 || e.children.length <= 2)
        );
        if (el) return el.textContent?.trim() ?? "";
      }
      return "unknown";
    });

    if (score === -1) {
      return { score: -1, label, error: "Could not parse score from page" };
    }

    return { score, label };
  } catch (err) {
    return {
      score: -1,
      label: "error",
      error: err instanceof Error ? err.message : String(err),
    };
  } finally {
    await browser.close();
  }
}
