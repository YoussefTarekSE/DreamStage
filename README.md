# DreamStage

**An AI producer for people who've never touched a DAW.**

DreamStage takes a voice — recorded straight from the browser, no equipment, no music
theory — and turns it into a fully produced, mixed, and mastered song. It listens to
*how* someone actually sings and builds an accompaniment around that specific
performance, instead of handing back a generic beat.

> Status: **pre-launch beta build**, solo-founder project, zero external budget.
> Live at [dreamstage.vercel.app](https://dreamstage.vercel.app) once deployed.

---

## Table of contents

- [What it does](#what-it-does)
- [The three laws](#the-three-laws)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Running the app](#running-the-app)
- [Testing](#testing)
- [Deployment](#deployment)
- [Configuration reference](#configuration-reference)
- [Roadmap](#roadmap)
- [License](#license)

---

## What it does

1. **Record** a vocal in the browser (Web Audio API + live waveform) — no gear needed.
2. **Clean it up** — noise reduction, EQ, compression, and pitch correction tuned to
   one of several vocal styles (Natural, Subtle, Modern Pop, R&B, Rap, Melodic, Heavy).
3. **Generate a beat that follows the vocal**, not a preset loop: the backend
   transcribes the singer's actual melody and chord progression, detects key and
   tempo, and builds bass, chords, drums, and lead lines that harmonize with what was
   sung — including a reactive arrangement that builds and drops with the performance.
4. **Explore freely with Producer Cuts** — every generation is kept forever (never a
   capped "3 tries and it's gone"). Artists can replay, favorite, restore, or *branch*
   a new direction from any past cut. The system learns from what gets favorited,
   accepted, or skipped, and leads future generations toward that taste.
5. **Get plain-language coaching** from an AI producer persona on pitch, timing, and
   delivery — never on what to say or how to sound like someone else.
6. **Mix and master** automatically to streaming-ready loudness (-14 LUFS), then
   export as MP3 (320 kbps) or WAV (24-bit/48kHz).

## The three laws

DreamStage's product decisions are constrained by three non-negotiable rules:

1. **AI assists, never replaces.** The artist's voice and creative choices are
   untouchable — DreamStage shapes and supports them, it doesn't rewrite them.
2. **AI explains every decision in plain language.** No opaque sliders or jargon;
   every suggestion comes with a reason a first-time artist can understand.
3. **Zero prior music knowledge required.** No music theory, no DAW experience, no
   equipment — a phone mic and a voice are enough.

## How it works

```
Browser recording
      │
      ▼
Vocal cleanup + pitch correction  (noisereduce, pedalboard, WORLD vocoder)
      │
      ▼
Vocal analysis: key, tempo, melody, chord progression, energy contour
      │
      ▼
Beat generation — tiered, always produces a result:
  1. ACE-Step 1.5 (neural, "Vocal2BGM"-style accompaniment over the real vocal) — when a GPU server is reachable
  2. HF-hosted MusicGen (text-to-music transformer)
  3. Local MusicGen (audiocraft, dev only)
  4. Programmatic synthesizer — real sampled instruments (drums, 808/pluck bass,
     SoundFont-rendered pads/leads) arranged from the vocal's own harmony, melody,
     and energy — always available, never fails
      │
      ▼
Producer Cuts — unlimited generations, kept forever, favorite / branch / restore,
taste-learning across the artist's history
      │
      ▼
AI Coach feedback (Claude) — plain-language notes on the take
      │
      ▼
Automatic mix + master (loudness staging, sidechain duck, harmony stacking,
multiband compression, tape saturation, streaming-safe -14 LUFS)
      │
      ▼
Export — MP3 320kbps / WAV 24-bit-48kHz — private project library
```

The guiding principle (the "vocal-first doctrine"): the singer's actual take drives
every downstream decision — key, chords, arrangement dynamics, and the melodic hook —
rather than the vocal being dropped onto a generic template.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 (App Router), React 19, Tailwind CSS 4, Zustand, TanStack Query, WaveSurfer.js → deployed on Vercel |
| Backend API | FastAPI (Python 3.12) → deployed on Render |
| Database / Auth | Supabase (PostgreSQL + Auth) |
| Object storage | Cloudflare R2 (S3-compatible) |
| Audio DSP | librosa, pyworld (WORLD vocoder), pedalboard, noisereduce, pyloudnorm, tinysoundfont |
| AI — beat generation | ACE-Step 1.5 (neural, MIT-licensed, optional local/GPU server), Hugging Face MusicGen, audiocraft (local dev) |
| AI — analysis | scikit-learn classifiers (genre / mood / energy, trained in-repo), CREPE / YIN pitch detection |
| AI — coaching | Anthropic Claude API |
| CI | GitHub Actions |

## Repository structure

```
dreamstage/
├── frontend/            Next.js app (recording UI, studio, producer cuts, auth)
├── backend/
│   ├── app/
│   │   ├── routers/     REST endpoints (projects, voice_training, studio, beat, coach, mix, admin)
│   │   ├── services/    Core logic: vocal processing, beat generation/synthesis/scoring,
│   │   │                mixing, ACE-Step client, producer cuts, ML analysis, telemetry
│   │   ├── assets/      Shipped sample libraries (drum kits, 808/pluck bass, texture loops)
│   │   └── models/      Pydantic request/response schemas
│   ├── ai/               Training pipeline for the genre/mood/energy classifiers
│   │                      (dataset download → feature extraction → training → models/*.joblib)
│   ├── supabase/         SQL migrations
│   └── tests/            Pytest suite
├── supabase/migrations/ Canonical migration history (applied via scripts/migrate.ps1)
├── scripts/migrate.ps1  Applies SQL migrations through the Supabase Management API
├── render.yaml           Render deployment config for the backend
├── DEPLOYMENT.md          Deployment notes
└── start.ps1              One-shot local dev launcher (Windows)
```

> **Note on ACE-Step:** the neural "listens to the real vocal and composes an
> accompaniment" tier is powered by [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)
> (MIT-licensed), run as a separate local/GPU server. It's not vendored in this repo —
> clone it independently if you want that tier locally:
> ```bash
> git clone https://github.com/ace-step/ACE-Step-1.5.git
> ```
> DreamStage works fully without it: when no ACE-Step server is reachable at
> `ACESTEP_URL`, beat generation transparently falls through to the next tier.

## Getting started

### Prerequisites

- Python 3.12
- Node.js 20+
- A [Supabase](https://supabase.com) project (free tier works)
- A Cloudflare R2 bucket (or any S3-compatible storage) for audio uploads
- Optional: a Hugging Face API key (beat generation), a Groq API key, and an
  Anthropic API key (AI coach)

### 1. Clone and configure

```bash
git clone https://github.com/<your-org>/dreamstage.git
cd dreamstage
cp .env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

Fill in `backend/.env` and `frontend/.env.local` with your Supabase, R2, and AI
provider credentials — see [Configuration reference](#configuration-reference).

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

### 3. Database

Apply the SQL migrations through the Supabase Management API:

```powershell
$env:SUPABASE_MANAGEMENT_TOKEN = "..."
$env:SUPABASE_PROJECT_REF = "..."
.\scripts\migrate.ps1
```

### 4. Frontend

```bash
cd frontend
npm ci
```

## Running the app

**Windows one-shot launcher** (checks prerequisites, creates the venv, installs
dependencies, and starts both servers):

```powershell
.\start.ps1
```

**Manual start:**

```bash
# Backend — http://localhost:8000
cd backend
uvicorn app.main:app --reload --port 8000

# Frontend — http://localhost:3000
cd frontend
npm run dev
```

`GET /health` reports the status of the app, database, storage, and AI subsystems.
`GET /metrics` reports in-process generation/mix/cache metrics.

## Testing

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm run lint
npx tsc --noEmit
npm run test
npm run build
```

CI runs on every push via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Deployment

- **Backend** deploys to [Render](https://render.com) from [`render.yaml`](render.yaml)
  (`rootDir: backend`, health check on `/health`). All secrets are set in the Render
  dashboard, never committed.
- **Frontend** deploys to [Vercel](https://vercel.com) from the `frontend/` directory.
- Full step-by-step notes live in [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Configuration reference

All required environment variables are documented in [`.env.example`](.env.example)
(shared/backend) and [`frontend/.env.example`](frontend/.env.example). Highlights:

| Variable | Purpose |
|---|---|
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWT_SECRET` | Database + auth |
| `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` | Audio file storage |
| `GROQ_API_KEY`, `HF_API_KEY` | Beat generation providers |
| `ADMIN_KEY` | Bearer token for `/admin/*` endpoints |
| `DAILY_PROJECT_LIMIT`, `DAILY_BEAT_GENERATION_LIMIT`, `DAILY_MIX_LIMIT` | Free-tier usage caps (0 disables) |
| `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_API_URL` | Frontend public config |

Never commit real `.env` files — they're gitignored, and only the `.env.example`
templates are tracked.

## Roadmap

Currently in the pre-beta build phase, working toward a 10-artist closed beta.

- **Shipped:** recording, vocal cleanup + 8 vocal styles, tiered beat generation with
  vocal-following harmony/melody/arrangement, unlimited Producer Cuts with taste
  learning, AI coach feedback, automatic mix/master, MP3/WAV export.
- **In progress:** ACE-Step neural accompaniment tier evaluation, async job handling
  for long-running generations, dedicated GPU worker.
- **Post-beta:** lyric assistant, melody suggestions, voice progress tracking, public
  artist profiles, streaming platform publishing, mobile app.

**Explicitly out of scope:** voice cloning (never), a social feed, a beat
marketplace, and AI-written lyrics at MVP.

## License

Proprietary — all rights reserved. This code is not licensed for reuse,
redistribution, or derivative works without explicit permission from the project
owner.

The vendored/optional [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)
integration is a separate project under the MIT License; see its own repository for
terms.
