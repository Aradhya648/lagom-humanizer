import { NextRequest, NextResponse } from "next/server";
import { humanizeDeep } from "@/lib/deep-humanizer";
import { type ContentType } from "@/prompts/pipeline";

// Long timeout — 4 browser sessions + 4 LLM calls can take up to 2 min
export const maxDuration = 300;

// Allow CORS from Vercel frontend
function corsHeaders(origin: string | null) {
  const allowed = process.env.ALLOWED_ORIGIN ?? "*";
  return {
    "Access-Control-Allow-Origin": allowed,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

export async function OPTIONS(req: NextRequest) {
  return new NextResponse(null, {
    status: 204,
    headers: corsHeaders(req.headers.get("origin")),
  });
}

export async function POST(req: NextRequest) {
  const origin = req.headers.get("origin");

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
      return NextResponse.json({ error: "text must be a non-empty string" }, {
        status: 400,
        headers: corsHeaders(origin),
      });
    }

    const validTypes: ContentType[] = ["essay", "academic", "email", "document", "general"];
    if (!validTypes.includes(contentType as ContentType)) {
      return NextResponse.json(
        { error: "contentType must be one of: essay, academic, email, document, general" },
        { status: 400, headers: corsHeaders(origin) }
      );
    }

    const limit = typeof wordLimit === "number" && wordLimit > 0 ? wordLimit : 1000;
    const thresh = typeof threshold === "number" ? threshold : 15;
    const maxIter = typeof maxIterations === "number" ? Math.min(maxIterations, 4) : 4;

    const result = await humanizeDeep(text, contentType as ContentType, limit, {
      threshold: thresh,
      maxIterations: maxIter,
    });

    return NextResponse.json({
      humanizedText: result.text,
      finalScores: result.finalScores,
      iterations: result.iterations,
      scoreHistory: result.scoreHistory,
    }, { headers: corsHeaders(origin) });
  } catch (error) {
    console.error("Humanize-deep API error:", error);
    const message = error instanceof Error ? error.message : "Internal server error";
    return NextResponse.json({ error: message }, {
      status: 500,
      headers: corsHeaders(origin),
    });
  }
}
