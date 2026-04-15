import { NextRequest, NextResponse } from "next/server";
import { humanize } from "@/lib/humanizer";
import { detectAI } from "@/lib/detector";
import { type ContentType } from "@/prompts/pipeline";

// Serverless runtime. Hobby plan caps at 10s.
// Pipeline uses gemini-2.0-flash (1-2s/call) to fit within that window.
export const maxDuration = 60;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { text, contentType, wordLimit } = body as {
      text: unknown;
      contentType: unknown;
      wordLimit: unknown;
    };

    // Validate text
    if (typeof text !== "string" || text.trim().length === 0) {
      return NextResponse.json(
        { error: "text must be a non-empty string" },
        { status: 400 }
      );
    }

    // Validate contentType
    const validTypes: ContentType[] = ["essay", "academic", "email", "document", "general"];
    if (!validTypes.includes(contentType as ContentType)) {
      return NextResponse.json(
        { error: "contentType must be one of: essay, academic, email, document, general" },
        { status: 400 }
      );
    }

    // Validate wordLimit
    const limit =
      typeof wordLimit === "number" && wordLimit > 0 && wordLimit <= 1000
        ? wordLimit
        : 1000;

    // Check word count
    const wordCount = text.trim().split(/\s+/).length;
    if (wordCount === 0) {
      return NextResponse.json(
        { error: "text cannot be empty" },
        { status: 400 }
      );
    }

    // Score original
    const originalResult = detectAI(text);

    // Humanize
    const humanizedText = await humanize(text, contentType as ContentType, limit);

    // Score humanized
    const humanizedResult = detectAI(humanizedText);

    return NextResponse.json({
      humanizedText,
      originalScore: originalResult.score,
      humanizedScore: humanizedResult.score,
    });
  } catch (error) {
    console.error("Humanize API error:", error);
    const message =
      error instanceof Error ? error.message : "Internal server error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
