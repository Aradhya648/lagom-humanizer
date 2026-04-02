# Lagom Humanizer

> *Just the right amount of human.*

An AI-powered text humanizer that transforms AI-generated writing into natural, human-sounding prose. Built for writers who want to beat AI detection without losing their message.

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

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | **Next.js 16** (App Router) |
| Language | **TypeScript** |
| Styling | **Tailwind CSS** |
| AI | **Google Gemini 2.5 Flash** (free tier) |
| Detection | Custom heuristic detector |

## Getting Started

```bash
git clone https://github.com/Aradhya648/lagom-humanizer.git
cd lagom-humanizer
npm install
cp .env.example .env.local
# Add your Gemini API key: https://aistudio.google.com/app/apikey
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Get it free at [Google AI Studio](https://aistudio.google.com/app/apikey) |

## API

```bash
POST /api/humanize
{
  "text": "Your text...",
  "mode": "medium",
  "wordLimit": 500
}
```

Response: `{ "humanizedText": "...", "originalScore": 92, "humanizedScore": 23 }`

## Deploy

```bash
npx vercel --prod
```

Live: [thelagom.vercel.app](https://thelagom.vercel.app)

*Lagom — built for writers, not robots.*