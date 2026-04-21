import { NextRequest, NextResponse } from "next/server";
import { humanizeDeep, type DeepEvent } from "@/lib/deep-humanizer";
import { type ContentType } from "@/prompts/pipeline";

// maxDuration is a Vercel-only hint. Set to 300 (hobby plan ceiling) so
// Vercel builds cleanly. In practice, deep mode routes to Railway via
// NEXT_PUBLIC_DEEP_API_URL — Railway runs as a persistent server so this
// limit never applies there.
export const maxDuration = 300;

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

  let body: { text?: unknown; contentType?: unknown; wordLimit?: unknown; threshold?: unknown; maxIterations?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON body" },
      { status: 400, headers: corsHeaders(origin) }
    );
  }

  const { text, contentType, wordLimit, threshold, maxIterations } = body;

  // Validation (fail fast before streaming starts)
  if (typeof text !== "string" || text.trim().length === 0) {
    return NextResponse.json(
      { error: "text must be a non-empty string" },
      { status: 400, headers: corsHeaders(origin) }
    );
  }

  const validTypes: ContentType[] = ["essay", "academic", "email", "document", "general"];
  if (!validTypes.includes(contentType as ContentType)) {
    return NextResponse.json(
      { error: "contentType must be one of: essay, academic, email, document, general" },
      { status: 400, headers: corsHeaders(origin) }
    );
  }

  const limit = typeof wordLimit === "number" && wordLimit > 0 ? wordLimit : 1000;
  const thresh = typeof threshold === "number" ? threshold : 5;
  const maxIter = typeof maxIterations === "number" ? Math.min(maxIterations, 8) : 6;

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      const send = (event: DeepEvent) => {
        try {
          const payload = `data: ${JSON.stringify(event)}\n\n`;
          controller.enqueue(encoder.encode(payload));
        } catch {
          // Controller already closed
        }
      };

      // Heartbeat: send a comment every 15s so the connection doesn't idle out
      const heartbeat = setInterval(() => {
        try {
          controller.enqueue(encoder.encode(`: heartbeat\n\n`));
        } catch {
          clearInterval(heartbeat);
        }
      }, 15_000);

      try {
        await humanizeDeep(
          text,
          contentType as ContentType,
          limit,
          { threshold: thresh, maxIterations: maxIter },
          send,
        );
      } catch (err) {
        console.error("Humanize-deep error:", err);
        const message = err instanceof Error ? err.message : "Internal server error";
        send({ type: "error", message });
      } finally {
        clearInterval(heartbeat);
        try {
          controller.close();
        } catch { /* already closed */ }
      }
    },
  });

  return new Response(stream, {
    headers: {
      ...corsHeaders(origin),
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
