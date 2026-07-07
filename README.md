---
title: Music Byy
emoji: 🎵
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Music Byy

A music discovery tool framed as a compatibility check.

Two people type a few of their favorite songs each. The backend downloads the 30-second previews, runs real Fourier-based audio analysis to fingerprint each song in a 53-dimensional feature space, and computes a compatibility score from the cosine similarity of the two listeners' average fingerprints. The actual product is the recommendation engine that sits underneath: it searches a cache of thousands of pre-analyzed songs and returns the ones whose audio fingerprints sit closest to the **geometric midpoint** of the two listeners — the sonic intersection of their tastes, almost always songs neither of them has heard.

Live at **[musicbyy.com](https://musicbyy.com)**.

## How it works

**Each song gets a real audio fingerprint.** When a song comes in for the first time, the backend hits Deezer for its 30-second preview MP3. The iTunes Search API acts as a full second catalog (~100M songs, no auth): it fills in previews Deezer is missing *and* handles songs Deezer hasn't indexed at all — iTunes-only tracks are cached under an `it:`-prefixed ID so the two ID spaces never collide. librosa loads the audio at 22 kHz mono and extracts:

- **Tempo** via beat tracking
- **Energy** from RMS amplitude, spectral centroid, rolloff, bandwidth, and zero-crossing rate
- **Timbre** as 13 MFCC means and standard deviations
- **Harmonic content** as 12 chroma values and 6 tonnetz components
- **Mode** (major vs. minor) via Krumhansl-Kessler key profile correlation

These features are normalized and packed into a 53-dim vector that uniquely represents how the song sounds. The vector is written to a Supabase Postgres cache — so the song is never analyzed twice across the entire history of the deployment.

**Each user's "musical fingerprint" is the average of their songs' vectors.** Compatibility is the weighted blend of three subspace cosine similarities:

- **Energy** (cosine on the 7 intensity dims) — 30% weight
- **Tempo** (gaussian on raw BPM averages) — 20% weight  
- **Mood** (cosine on the 45 timbre/harmonic/mode dims) — 50% weight

Scaled to 0–100. A subscore breakdown is surfaced in the UI.

**Recommendations come from nearest-neighbor search in audio-feature space.** The recommender computes the midpoint vector between the two users and finds the closest cached songs via a **pgvector HNSW index inside Postgres** — memory-flat and fast whether the pool holds 5k or 1M songs. (If the pgvector extension is unavailable, it falls back to an in-memory numpy matrix refreshed every 5 minutes.) The top 6 with unique artists, excluding the input songs, are returned — each with a **match percentage** (how close it sits to the couple's midpoint, on the same perceptual scale as the compatibility score) and a fresh **30-second preview URL** so the frontend can play recommendations inline. The bigger the cache, the richer the discovery — quality scales with the size of the pre-warmed pool.

## Why this is different from Spotify Blend

Existing two-user compatibility tools (Spotify Blend, MusicTaste.space, etc.) use **collaborative filtering** on listening history — "people whose play patterns look like yours also play X." That requires a streaming-platform account, weeks of listening data, and access to a recommendation engine trained on millions of users.

Music Byy uses **content-based audio analysis** — "song X sounds acoustically similar to your shared midpoint." It works for anyone who can name 5 songs, on any platform, with no listening history required. No login. Cross-platform. And because the fingerprints come from raw signal analysis, the algorithm can surface genuinely surprising recommendations across genres that collaborative-filter engines wouldn't connect.

## Tech stack

| Layer | What it is |
| --- | --- |
| Frontend | Vanilla HTML / CSS / JS single file, hosted on GitHub Pages |
| Backend | FastAPI on Hugging Face Spaces (Docker) |
| Audio analysis | librosa + numpy + scipy |
| Cache / metadata store | Supabase Postgres via asyncpg, pgvector HNSW for nearest-neighbor search |
| Music data sources | Deezer Public API (primary), iTunes Search API (second catalog: search + preview fallback) |

### API endpoints

| Endpoint | What it does |
| --- | --- |
| `POST /match` | The whole product: search + analyze both people's songs, score compatibility, return midpoint recommendations with previews |
| `GET /suggest?q=` | Autocomplete merging Deezer + iTunes catalogs concurrently; never 500s (degrades to an empty list) |
| `GET /health` | Liveness probe + cached-track count |

The frontend is a single dependency-free file: animated score ring, per-category breakdown bars, inline 30-second preview playback on recommendations, keyboard-navigable autocomplete (↑ ↓ Enter Esc), staged loading messages that mirror the real pipeline, and Web Share / clipboard sharing of results.

## Project structure

```
app.py          FastAPI entry point, route handlers, lifespan
analyzer.py     librosa-based audio feature extraction + vector packing
deezer.py       Deezer search + cache-first track enrichment
itunes.py       iTunes Search API fallback for missing previews
match.py        Compatibility scoring + midpoint nearest-neighbor recommendations
cache.py        Supabase Postgres feature cache (async via asyncpg)
client.py       Shared httpx async client singleton
prewarm.py      One-shot script that walks Deezer's genres and fills the cache
index.html      The entire frontend in one file
Dockerfile      Hugging Face Spaces build instructions (installs ffmpeg + librosa deps)
requirements.txt Python deps pinned to Python 3.11 + 3.13 compatible versions
CNAME           GitHub Pages custom domain (musicbyy.com)
```

## Running locally

Prerequisites: Python 3.11+ and a Supabase Postgres project for the feature cache.

```bash
git clone https://github.com/yerosen-daba/musicbyy.git
cd musicbyy
pip install -r requirements.txt
export DATABASE_URL="postgresql://postgres.<project-id>:<password>@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
uvicorn app:app --reload --port 8000
```

Then open `index.html` in your browser. The frontend's `API` constant points at the live Hugging Face Space by default — change it to `http://localhost:8000` for fully local development.

### Pre-warming the cache

The midpoint recommender draws candidates entirely from the cache, so a populated cache is what makes recommendations rich. Run once after deploying:

```bash
python3 prewarm.py --tracks 5000 --concurrency 4
```

This walks seven discovery sources — genre charts, editorial sections, radio stations, per-year searches, a ~90-keyword playlist crawl, full album-discography walks of cached artists, and a cache-seeded artist expansion — dedupes, and writes feature vectors to Supabase. Roughly 1 second per track. Already-cached tracks are skipped on re-runs (interrupt and re-run freely), and `--sources playlists,albums` runs subsets.

For growing the pool toward a million songs, use turbo mode:

```bash
python3 prewarm.py --turbo
```

Turbo runs the three deep sources (playlists → album discographies → related-artist graph), analyzes 8 tracks in parallel, caps at 200k tracks per run, prints a rate/ETA line every 200 tracks, and backs off automatically when it hits Deezer's request quota. Each re-run seeds from a bigger cache, so successive overnight runs compound. Note: past ~100–150k tracks you'll outgrow Supabase's free-tier storage — the pgvector search itself is built to handle millions.

## Credits

The original prototype was built as a CS course project using metadata proxies (popularity rank, release year) as stand-ins for energy and mood. The current architecture — real Fourier-based audio analysis, Supabase-backed feature cache, midpoint nearest-neighbor recommendations, and the refreshed frontend — was developed in collaboration with Claude (Anthropic) over an extended pair-programming session.

Music data is provided by the Deezer Public API and the iTunes Search API.
