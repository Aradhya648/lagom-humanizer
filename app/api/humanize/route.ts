import { NextRequest, NextResponse } from "next/server";
import { humanize, HumanizeMode } from "@/lib/humanizer";
import { detectAI } from "@/lib/detector";

// Increase serverless function timeout for multi-pass aggressive mode.
// Vercel Pro/Enterprise honours this; Hobby plan caps at 10s regardless.
export const maxDuration = 60;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { text, mode, wordLimit } = body as {
      text: unknown;
      mode: unknown;
      wordLimit: unknown;
    };

    // Validate text
    if (typeof text !== "string" || text.trim().length === 0) {
      return NextResponse.json(
        { error: "text must be a non-empty string" },
        { status: 400 }
      );
    }

    // Validate mode
    const validModes: HumanizeMode[] = ["light", "medium", "aggressive"];
    if (!validModes.includes(mode as HumanizeMode)) {
      return NextResponse.json(
        { error: 'mode must be one of: light, medium, aggressive' },
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
    const humanizedText = await humanize(text, mode as HumanizeMode, limit);

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
