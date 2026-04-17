import { NextResponse } from "next/server";

export async function GET() {
  // Never expose the actual key value — just check if it's set
  return NextResponse.json({
    hasGeminiKey: !!process.env.GEMINI_API_KEY,
    keyLength: process.env.GEMINI_API_KEY?.length ?? 0,
    nodeEnv: process.env.NODE_ENV,
  });
}
