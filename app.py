"""
app.py — Music Byy Backend

Main entry point for the FastAPI server. Handles routing and HTTP responses.
"""

import asyncio
import struct
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

import cache
import client
import itunes
from deezer import DEEZER_SEARCH, search_many_tracks
from match import compute_compatibility, compatibility_message, get_recommendations


def _make_silent_wav(seconds: float = 6.0, sr: int = 22050) -> bytes:
    """Build a tiny silent WAV in memory. Used for librosa JIT warmup."""
    n_samples = int(seconds * sr)
    data = b"\x00\x00" * n_samples
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
        + b"data" + struct.pack("<I", len(data))
    )
    return header + data


@asynccontextmanager
async def lifespan(app):
    # ── HTTP client (shared, pooled connections) ────────────────────────────
    client.http_client = httpx.AsyncClient(timeout=30.0)

    # ── Postgres connection pool for the feature cache ──────────────────────
    # Reads DATABASE_URL from env. Set this to your Supabase Session Pooler URI.
    await cache.init_pool()

    # ── librosa warmup ──────────────────────────────────────────────────────
    # The first librosa call in a process eats ~12s of numba JIT compilation.
    # We trigger that here on boot (in the background, so it doesn't delay
    # readiness) instead of letting it land on the first real user request.
    async def _warm_librosa():
        try:
            import analyzer
            silent_wav = _make_silent_wav()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, analyzer._analyze_audio_bytes_sync, silent_wav)
        except Exception:
            pass  # warmup failure is non-fatal
    asyncio.create_task(_warm_librosa())

    yield

    # ── Shutdown ────────────────────────────────────────────────────────────
    await client.http_client.aclose()
    await cache.close_pool()

app = FastAPI(title="Music Byy API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

class MatchRequest(BaseModel):
    person1_name:  str = Field(min_length=1, max_length=40)
    person1_songs: list[str] = Field(min_length=1, max_length=10)
    person2_name:  str = Field(min_length=1, max_length=40)
    person2_songs: list[str] = Field(min_length=1, max_length=10)


def _clean_queries(songs: list[str]) -> list[str]:
    """Trim, drop empties, dedupe (case-insensitive, order-preserving)."""
    seen: set[str] = set()
    out: list[str] = []
    for s in songs:
        q = s.strip()[:200]
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out

# ─── Routes ───────────────────────────────────────────────────────────────────

def _strip_internal_fields(track: dict) -> dict:
    """Remove heavy/internal fields before sending a track over the wire.

    The frontend only needs metadata (name, artist, image, etc.); the 53-dim
    vector and full feature dict are server-internal — sending them would
    bloat the JSON payload for no reason.
    """
    drop = {"features", "vector", "preview", "deezer_id"}
    return {k: v for k, v in track.items() if k not in drop}


@app.post("/match")
async def match(data: MatchRequest):
    queries1 = _clean_queries(data.person1_songs)
    queries2 = _clean_queries(data.person2_songs)
    if not queries1 or not queries2:
        raise HTTPException(status_code=400, detail="Both people need at least one song.")

    tracks1, tracks2 = await asyncio.gather(
        search_many_tracks(queries1),
        search_many_tracks(queries2),
    )
    if not tracks1 or not tracks2:
        raise HTTPException(status_code=400, detail="Couldn't find songs. Try different names.")

    score           = compute_compatibility(tracks1, tracks2)
    recommendations = await get_recommendations(tracks1, tracks2)

    return {
        "person1_name":       data.person1_name,
        "person2_name":       data.person2_name,
        "score":              score["total"],
        "overall_similarity": score["overall_similarity"],
        "message":            compatibility_message(score["total"]),
        "vibe1":              score["vibe1"],
        "vibe2":              score["vibe2"],
        "energy_score":       score["energy_score"],
        "tempo_score":        score["tempo_score"],
        "mood_score":         score["mood_score"],
        # Keep valence_score as an alias for one release so any cached
        # frontend version still renders something; remove later.
        "valence_score":      score["mood_score"],
        "songs1":             [_strip_internal_fields(t) for t in tracks1],
        "songs2":             [_strip_internal_fields(t) for t in tracks2],
        "recommendations":    recommendations,
        "details":            score["details"],
    }

async def _deezer_suggest(q: str, limit: int) -> list[dict]:
    """Deezer autocomplete candidates. Never raises."""
    try:
        r = await client.http_client.get(
            DEEZER_SEARCH, params={"q": q, "limit": limit}, timeout=6.0,
        )
        results = r.json().get("data", []) or []
    except Exception:
        return []
    return [
        {
            "name":   t.get("title", ""),
            "artist": t.get("artist", {}).get("name", ""),
            "image":  t.get("album", {}).get("cover_small", ""),
        }
        for t in results
        if t.get("title")
    ]


@app.get("/suggest")
async def suggest(q: str, limit: int = 7):
    """Autocomplete backed by Deezer AND Apple's catalog, merged.

    Both searches run concurrently, so the union costs no extra latency.
    Deezer results rank first (its previews are our primary source); iTunes
    fills in tracks Deezer doesn't index. Never 500s — either source
    failing just narrows the list.
    """
    if not q or len(q.strip()) < 2:
        return []
    q = q.strip()[:200]
    limit = min(max(limit, 1), 12)

    deezer_results, itunes_results = await asyncio.gather(
        _deezer_suggest(q, limit),
        itunes.suggest(q, limit=5),
    )

    seen: set[tuple[str, str]] = set()
    out = []
    for t in deezer_results + itunes_results:
        key = (t["name"].lower(), t["artist"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= limit:
            break
    return out

@app.get("/health")
async def health():
    """Liveness + cache-size probe. Cheap enough to poll."""
    try:
        cached = await cache.count_cached()
    except Exception:
        cached = -1
    return {"status": "ok", "cached_tracks": cached}

@app.get("/")
async def home():
    return {"message": "Music Byy API is running", "docs": "/docs"}
