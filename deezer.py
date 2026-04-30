import asyncio
import client
from audio import analyze_preview

DEEZER_SEARCH = "https://api.deezer.com/search"
DEEZER_TRACK  = "https://api.deezer.com/track"

# Maps Deezer track ID → dict with audio features so we don't re-download
_feature_cache: dict[int, dict] = {}

# Limit concurrent CPU-heavy librosa threads to 4 to utilize the upgraded CPU
_analysis_semaphore = None

def get_semaphore():
    global _analysis_semaphore
    if _analysis_semaphore is None:
        _analysis_semaphore = asyncio.Semaphore(4)
    return _analysis_semaphore

async def search_track(query: str) -> dict | None:
    """Search Deezer for a track and return basic metadata."""
    r = await client.http_client.get(DEEZER_SEARCH, params={
        "q": query, "limit": 1,
    })
    data = r.json().get("data", [])
    if not data:
        return None

    t = data[0]
    track_id = t.get("id")
    artist = t.get("artist", {})
    album = t.get("album", {})

    return {
        "name":       t.get("title", ""),
        "artist":     artist.get("name", ""),
        "artist_id":  str(artist.get("id", "")),
        "track_id":   str(track_id),
        "deezer_id":  track_id,
        "image":      album.get("cover_medium", "") or album.get("cover_big", ""),
        "url":        t.get("link", ""),
        "preview":    t.get("preview", ""),
    }

async def enrich_track(track: dict) -> dict:
    """
    Fetch full track details from Deezer (BPM, preview URL, release date)
    then analyze the preview audio with librosa for energy/tempo/valence.
    """
    deezer_id = track.get("deezer_id")

    # Check cache first
    if deezer_id and deezer_id in _feature_cache:
        return {**track, **_feature_cache[deezer_id]}

    # Fetch full track details for BPM and reliable preview URL
    detail = {}
    if deezer_id:
        r = await client.http_client.get(f"{DEEZER_TRACK}/{deezer_id}")
        if r.status_code == 200:
            detail = r.json()

    # Get the preview URL (prefer detail endpoint, fall back to search result)
    preview_url = detail.get("preview") or track.get("preview", "")
    deezer_bpm  = detail.get("bpm", 0) or 0
    release_date = detail.get("release_date", "")

    features = {"energy": 0.5, "tempo": 120.0, "valence": 0.5}

    if preview_url:
        try:
            # Download the 30-second preview
            audio_r = await client.http_client.get(preview_url)
            if audio_r.status_code == 200 and len(audio_r.content) > 1000:
                # Use a semaphore to prevent CPU thrashing on Render's weak CPU
                sem = get_semaphore()
                async with sem:
                    features = await asyncio.to_thread(analyze_preview, audio_r.content)

                # Cross-check tempo with Deezer's BPM if available
                if deezer_bpm > 0:
                    # If librosa and Deezer agree (within 20%), use librosa.
                    # Otherwise, prefer Deezer's BPM (may be more reliable for
                    # songs with complex rhythms)
                    ratio = features["tempo"] / deezer_bpm if deezer_bpm else 1.0
                    if ratio < 0.8 or ratio > 1.2:
                        features["tempo"] = deezer_bpm
        except Exception:
            pass  # Fall through to defaults

    enriched = {
        **track,
        "energy":       features["energy"],
        "tempo":        features["tempo"],
        "valence":      features["valence"],
        "release_date": release_date,
    }

    # Cache the features
    if deezer_id:
        _feature_cache[deezer_id] = {
            "energy":       features["energy"],
            "tempo":        features["tempo"],
            "valence":      features["valence"],
            "release_date": release_date,
        }

    return enriched

async def search_and_enrich(query: str) -> dict | None:
    """Search for a track, then enrich it with audio features."""
    track = await search_track(query)
    if not track:
        return None
    return await enrich_track(track)

async def search_many_tracks(queries: list[str]) -> list[dict]:
    """Search and analyze multiple tracks in parallel."""
    results = await asyncio.gather(*[search_and_enrich(q) for q in queries])
    return [r for r in results if r is not None]
