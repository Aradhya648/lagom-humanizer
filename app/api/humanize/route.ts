import { NextRequest, NextResponse } from 'next/server'
import { validateInput } from '@/lib/inputValidator'
import { runRewritePipeline } from '@/lib/rewrite/pipeline'
import type { WritingMode } from '@/lib/rewrite/types'

const VALID_MODES: WritingMode[] = ['academic', 'casual', 'formal']
const DEV_MODE = process.env.DEV_MODE === 'true'

export async function POST(req: NextRequest) {

  // ── ⬡ RATE LIMIT INSERTION POINT ──────────────────────────────────
  // import { checkRateLimit } from '@/lib/rateLimit'
  // const ip = req.headers.get('x-forwarded-for') ?? 'unknown'
  // const allowed = await checkRateLimit(ip)
  // if (!allowed) return NextResponse.json({ error: 'Rate limit exceeded' }, { status: 429 })
  // ──────────────────────────────────────────────────────────────────

  try {
    const body = await req.json()
    const { text, mode } = body as { text?: string; mode?: string }

    if (!mode || !VALID_MODES.includes(mode as WritingMode)) {
      return NextResponse.json(
        { error: `Invalid mode. Must be one of: ${VALID_MODES.join(', ')}` },
        { status: 400 }
      )
    }

    const validation = validateInput(text ?? '')
    if (!validation.valid) {
      return NextResponse.json({ error: validation.error }, { status: 400 })
    }

    const result = await runRewritePipeline({
      text: text!.trim(),
      mode: mode as WritingMode,
    })

    return NextResponse.json({
      final: result.final,
      metrics: result.metrics,
      ...(DEV_MODE && {
        dev: {
          passA: result.passA,
          passB: result.passB,
          passC: result.passC,
        },
      }),
    })

  } catch (err) {
    console.error('[/api/humanize] Error:', err)
    return NextResponse.json(
      { error: 'Something went wrong. Please try again.' },
      { status: 500 }
    )
  }
}
