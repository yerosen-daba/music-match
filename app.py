"""
app_v2.py — Music Match Backend (Modular Edition)

Main entry point for the FastAPI server.
"""

import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import client
from deezer import DEEZER_SEARCH, search_many_tracks
from match import compute_compatibility, compatibility_message, get_recommendations

@asynccontextmanager
async def lifespan(app):
    client.http_client = httpx.AsyncClient(timeout=30.0)
    yield
    await client.http_client.aclose()

app = FastAPI(title="Music Match v2 (Modular)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    person1_name:  str
    person1_songs: list[str]
    person2_name:  str
    person2_songs: list[str]

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/match")
async def match(data: MatchRequest):
    tracks1, tracks2 = await asyncio.gather(
        search_many_tracks(data.person1_songs),
        search_many_tracks(data.person2_songs),
    )
    if not tracks1 or not tracks2:
        raise HTTPException(status_code=400, detail="Couldn't find songs. Try different names.")

    score           = compute_compatibility(tracks1, tracks2)
    recommendations = await get_recommendations(tracks1, tracks2)

    return {
        "person1_name":    data.person1_name,
        "person2_name":    data.person2_name,
        "score":           score["total"],
        "message":         compatibility_message(score["total"]),
        "vibe1":           score["vibe1"],
        "vibe2":           score["vibe2"],
        "energy_score":    score["energy_score"],
        "tempo_score":     score["tempo_score"],
        "valence_score":   score["valence_score"],
        "songs1":          tracks1,
        "songs2":          tracks2,
        "recommendations": recommendations,
        "details":         score["details"],
    }

@app.get("/suggest")
async def suggest(q: str, limit: int = 7):
    """Deezer-backed autocomplete. No auth required."""
    if not q or len(q.strip()) < 2:
        return []
    r = await client.http_client.get(DEEZER_SEARCH, params={
        "q": q, "limit": min(limit, 10),
    })
    results = r.json().get("data", []) or []
    return [
        {
            "name":   t.get("title", ""),
            "artist": t.get("artist", {}).get("name", ""),
            "image":  t.get("album", {}).get("cover_small", ""),
        }
        for t in results
    ]

@app.get("/")
async def home():
    return {"message": "Music Match API v2 (Modular) is running"}
