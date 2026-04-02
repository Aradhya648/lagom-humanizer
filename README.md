# Lagom Humanizer

> *Just the right amount of human.*

An AI-powered text humanizer that transforms AI-generated writing into natural, human-sounding prose. Built for writers who want to beat AI detection without losing their message.

---

## Features

- **Three humanization modes**
  - **Light** — subtle refinements, preserves original voice
  - **Medium** — balanced restructuring and word choice
  - **Aggressive** — full rewrite for maximum humanization

- **Real-time AI detection** — heuristic-based scoring runs live as you type (600ms debounce)
- **Before/after comparison** — see your score improvement
- **Adjustable word limit** — slider from 100 to 1000 words
- **Live status indicator** — shows when Gemini API is connected
- **One-click copy** — grab your humanized text instantly

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | **Next.js 16** (App Router) |
| Language | **TypeScript** |
| Styling | **Tailwind CSS** |
| AI | **Google Gemini 2.5 Flash** (free tier) |
| Detection | Custom heuristic detector |

---

## Architecture

### High-level Data Flow

```
User Input (Text) → Live Detection (Client) → API: /api/humanize → Gemini API → Humanized Text + Scores → UI
```

### Folder Structure

```
lagom-humanizer/
├── app/
│   ├── api/
│   │   └── humanize/
│   │       └── route.ts      # POST endpoint: validates, calls humanize(), returns JSON
│   ├── globals.css           # Global styles + Tailwind
│   ├── layout.tsx            # Root layout
│   └── page.tsx              # Main UI (two-panel)
├── components/
│   ├── ModeSelector.tsx      # Light/Medium/Aggressive toggle
│   ├── ScorePill.tsx         # AI score badge (color-coded)
│   ├── WordReveal.tsx        # Word count animation
│   └── Spinner.tsx           # Loading spinner
├── lib/
│   ├── detector.ts           # AI detection heuristic
│   └── humanizer.ts          # Gemini client + prompt engineering
├── .env.example              # Environment template
├── package.json
├── tsconfig.json
└── README.md
```

### Key Components

#### `app/page.tsx`
- Client component (use client)
- Manages state: inputText, outputText, mode, wordLimit, loading scores
- Debounced AI detection on input (600ms)
- Calls `/api/humanize` with mode and wordLimit
- Displays two-panel layout (input / output) on large screens
- Shows ScorePill for original and humanized scores
- Copy-to-clipboard button

#### `app/api/humanize/route.ts`
- Server-side API route (Next.js App Router)
- Validates: text non-empty, mode ∈ {light,medium,aggressive}, wordLimit ≤ 1000
- Runs `detectAI(text)` for originalScore
- Calls `humanize(text, mode, limit)` from lib/humanizer.ts
- Runs `detectAI(humanizedText)` for humanizedScore
- Returns `{ humanizedText, originalScore, humanizedScore }`
- Sets `maxDuration = 60` for Vercel Pro (aggressive multi-pass)

#### `lib/detector.ts`
- Heuristic-based AI detection (non-LLM)
- Analyzes features: sentence uniformity, vocabulary diversity, syntactic patterns
- Returns `{ score: number (0-100), label: "Low" | "Medium" | "High" }`
- Runs entirely client-side for instant feedback

#### `lib/humanizer.ts`
- Uses Google Generative AI SDK (`@google/generative-ai`)
- Configures `genAI` with `GEMINI_API_KEY`
- Model: `gemini-2.5-flash`
- Constructs mode-specific system prompts:
  - Light: "Rephrase to sound more human, minimal changes"
  - Medium: "Rewrite naturally, vary sentence structure"
  - Aggressive: "Complete rewrite, break AI patterns, use contractions, idioms"
- Respects `wordLimit` by truncating output if needed

---

## Getting Started

### Prerequisites

- Node.js 18+
- Google Gemini API key (free tier): https://aistudio.google.com/app/apikey

### Local Development

```bash
# Clone
git clone https://github.com/Aradhya648/lagom-humanizer.git
cd lagom-humanizer

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local and set:
# GEMINI_API_KEY=your_key_here

# Run dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key (free tier) |

Get your key at: https://aistudio.google.com/app/apikey

---

## Usage

1. Paste AI-generated text into the input panel.
2. Choose a humanization mode (Light / Medium / Aggressive).
3. Adjust the word limit slider if needed (100–1000 words).
4. Click **Humanize**.
5. View the output score and copy the result.

The live detector shows AI likelihood as you type. After humanizing, compare the score drop.

---

## API Reference

### POST `/api/humanize`

**Request**

```json
{
  "text": "string (non-empty)",
  "mode": "light | medium | aggressive",
  "wordLimit": 1000  // optional, default 1000, max 1000
}
```

**Response**

```json
{
  "humanizedText": "string",
  "originalScore": 0-100,
  "humanizedScore": 0-100
}
```

The score is a heuristic AI likelihood (0 = human-like, 100 = AI-like).

---

## Deployment

### Vercel (recommended)

```bash
npx vercel --prod
```

Set environment variable `GEMINI_API_KEY` in Vercel project settings.

### Other hosts

Ensure Node.js 18+ and run:

```bash
npm run build
npm start
```

Configure `GEMINI_API_KEY` in your hosting environment.

---

## Benchmarks

The `benchmarks/` folder contains scripts to test detection and humanization accuracy across different text types.

---

## Future Improvements

- Support for multiple AI providers (OpenAI, Claude)
- Batch processing
- Firefox/Chrome extension
- Local LLM fallback (privacy mode)
- User accounts and history
- Custom mode creation

---

## License

Private project — all rights reserved.

---

## Contact

Live app: https://lagom-one.vercel.app

GitHub: https://github.com/Aradhya648/lagom-humanizer

---

*Lagom — built for writers, not robots.*
