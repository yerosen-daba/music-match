import asyncio
import client

DEEZER_SEARCH = "https://api.deezer.com/search"
DEEZER_TRACK  = "https://api.deezer.com/track"

# Maps Deezer track ID → dict with audio features so we don't re-download
_feature_cache: dict[int, dict] = {}

# Limit concurrent CPU-heavy librosa threads to 4 to utilize the upgraded CPU
_analysis_semaphore = None

def get_semaphore():
    global _analysis_semaphore
    if _analysis_semaphore is None:
        _analysis_semaphore = asyncio.Semaphore(3)
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
    Fetch full track details from Deezer (BPM, rank, release date).
    Uses metadata proxies for energy and valence to maximize performance.
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

    # Bypass librosa MP3 downloads entirely to guarantee < 5 second response times on Render.
    # Proxy 'energy' from song popularity rank, keep tempo, proxy 'valence' from release era.
    rank = detail.get("rank", 500000)
    features["energy"] = min(rank / 1000000.0, 1.0)
    
    if deezer_bpm > 0:
        features["tempo"] = float(deezer_bpm)
        
    if release_date:
        try:
            year = int(release_date.split("-")[0])
            features["valence"] = min(max((year - 1970) / 55.0, 0.0), 1.0)
        except Exception:
            pass

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
