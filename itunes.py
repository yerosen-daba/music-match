"""
itunes.py — iTunes Search API: second catalog + preview fallback.

Two roles:
  1. Preview fallback — when Deezer indexes a track but has no 30-second
     preview, we grab Apple's preview instead (find_preview_url).
  2. Full search fallback — when Deezer's search misses a track entirely,
     we search Apple's ~100M-song catalog and analyze its preview
     (search_track). This roughly doubles the songs users can type.

The API is free, requires no auth, and previews are ~30s m4a files on
Apple's CDN (librosa handles m4a via ffmpeg, installed in the Dockerfile).

iTunes-sourced tracks are cached under IDs prefixed "it:" so they can never
collide with Deezer's numeric ID space.

API reference: https://performance-partners.apple.com/search-api
"""

import client

ITUNES_SEARCH = "https://itunes.apple.com/search"
ITUNES_LOOKUP = "https://itunes.apple.com/lookup"
ITUNES_TIMEOUT = 5.0  # seconds — keep tight; this is a fallback

# Cache-key prefix marking a track as living in Apple's ID space.
ID_PREFIX = "it:"


async def _search_songs(term: str, limit: int) -> list[dict]:
    """Raw song search against the iTunes Search API. [] on any failure."""
    if not term:
        return []
    try:
        r = await client.http_client.get(
            ITUNES_SEARCH,
            params={"term": term, "media": "music", "entity": "song", "limit": limit},
            timeout=ITUNES_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        return r.json().get("results", []) or []
    except Exception:
        return []


def _artwork(t: dict) -> str:
    # iTunes returns 100x100 art; the URL pattern scales to any size.
    return (t.get("artworkUrl100") or "").replace("100x100", "250x250")


async def search_track(query: str) -> dict | None:
    """Search Apple's catalog for a track. Same dict shape as
    deezer.search_track, so downstream enrichment/caching works unchanged.
    """
    results = await _search_songs(query, limit=1)
    if not results:
        return None

    t = results[0]
    track_id = t.get("trackId")
    if not track_id:
        return None

    return {
        "name":      t.get("trackName", ""),
        "artist":    t.get("artistName", ""),
        "artist_id": f"{ID_PREFIX}{t.get('artistId', '')}",
        "track_id":  f"{ID_PREFIX}{track_id}",
        "deezer_id": f"{ID_PREFIX}{track_id}",  # cache key in Apple ID space
        "image":     _artwork(t),
        "url":       t.get("trackViewUrl", ""),
        "preview":   t.get("previewUrl", ""),
    }


async def suggest(query: str, limit: int = 5) -> list[dict]:
    """Autocomplete candidates from Apple's catalog. Never raises."""
    results = await _search_songs(query, limit=limit)
    return [
        {
            "name":   t.get("trackName", ""),
            "artist": t.get("artistName", ""),
            "image":  (t.get("artworkUrl60") or t.get("artworkUrl100") or ""),
        }
        for t in results
        if t.get("trackName")
    ]


async def get_preview_url(itunes_id: str) -> str:
    """Fresh preview URL for a cached iTunes track (bare ID, no prefix)."""
    if not itunes_id:
        return ""
    try:
        r = await client.http_client.get(
            ITUNES_LOOKUP, params={"id": itunes_id}, timeout=ITUNES_TIMEOUT,
        )
        results = r.json().get("results", []) or []
        return (results[0].get("previewUrl") or "") if results else ""
    except Exception:
        return ""


async def find_preview_url(track_name: str, artist_name: str) -> str | None:
    """Search iTunes for a track and return its 30-second preview URL.

    Returns None if nothing found, the request fails, or the matched track
    has no preview (rare but possible).

    The preview is typically an m4a file ~30 seconds long on Apple's CDN.
    librosa handles m4a fine as long as ffmpeg is installed on the host
    (which we install via the Dockerfile).
    """
    if not track_name:
        return None

    # Build a query string. Including the artist disambiguates covers and
    # remixes. iTunes ranks by relevance so the first match is almost always
    # the correct one when artist is included.
    query = f"{track_name} {artist_name}".strip()

    try:
        r = await client.http_client.get(
            ITUNES_SEARCH,
            params={
                "term":   query,
                "media":  "music",
                "entity": "song",
                "limit":  1,
            },
            timeout=ITUNES_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    results = data.get("results", [])
    if not results:
        return None

    return results[0].get("previewUrl") or None
