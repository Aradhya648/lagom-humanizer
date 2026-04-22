import { NextRequest, NextResponse } from "next/server";
import { sendErrorEmail } from "@/lib/mailer";

export async function POST(req: NextRequest) {
  try {
    const { technical, context, userFacing } = await req.json();
    await sendErrorEmail(
      String(technical ?? "unknown"),
      String(context ?? "unknown"),
      String(userFacing ?? "unknown")
    );
  } catch {
    // Never let error reporting break the app
  }
  return NextResponse.json({ ok: true });
}
