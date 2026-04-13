import { NextRequest, NextResponse } from "next/server";
import { humanizeLoop } from "@/lib/humanizer";
import { type ContentType } from "@/prompts/pipeline";

// Long timeout — Playwright + multiple LLM iterations can take 60–90s
export const maxDuration = 120;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { text, contentType, wordLimit, threshold, maxIterations } = body as {
      text: unknown;
      contentType: unknown;
      wordLimit: unknown;
      threshold: unknown;
      maxIterations: unknown;
    };

    if (typeof text !== "string" || text.trim().length === 0) {
      return NextResponse.json({ error: "text must be a non-empty string" }, { status: 400 });
    }

    const validTypes: ContentType[] = ["essay", "academic", "email", "document", "general"];
    if (!validTypes.includes(contentType as ContentType)) {
      return NextResponse.json(
        { error: "contentType must be one of: essay, academic, email, document, general" },
        { status: 400 }
      );
    }

    const limit = typeof wordLimit === "number" && wordLimit > 0 ? wordLimit : 1000;
    const thresh = typeof threshold === "number" ? threshold : 15;
    const maxIter = typeof maxIterations === "number" ? Math.min(maxIterations, 3) : 3;

    const result = await humanizeLoop(text, contentType as ContentType, limit, {
      threshold: thresh,
      maxIterations: maxIter,
    });

    return NextResponse.json({
      humanizedText: result.text,
      gptzeroScore: result.gptzeroScore,
      iterations: result.iterations,
      scoreHistory: result.scoreHistory,
    });
  } catch (error) {
    console.error("Humanize-loop API error:", error);
    const message = error instanceof Error ? error.message : "Internal server error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
