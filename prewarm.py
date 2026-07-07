"""
prewarm.py — One-shot script to populate the feature cache with a broad,
diverse pool of analyzed songs.

The midpoint-search recommender draws candidates exclusively from the cache,
so cache size + diversity directly determines how good recommendations are.
This script walks several Deezer discovery sources to maximize coverage:

  1. Genre charts          — top tracks per genre (~150 genres)
  2. Editorial sections    — Deezer's curated playlists per region/mood
  3. Radio stations        — algorithmic per-genre radio (~250 stations)
  4. Per-year searches     — historical catalog 1960–2025

With default settings it discovers ~10,000–15,000 unique tracks across
sources, deduped. Already-cached tracks are skipped on re-runs.

Usage:
    DATABASE_URL=... python3 prewarm.py
    DATABASE_URL=... python3 prewarm.py --tracks 8000
    DATABASE_URL=... python3 prewarm.py --concurrency 4
    DATABASE_URL=... python3 prewarm.py --sources genres,radio   # subset
"""

import argparse
import asyncio
import os
import sys
import time

import httpx

# Make our modules importable when run from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cache
import client
import deezer


# ─── Defaults ───────────────────────────────────────────────────────────────

PER_GENRE_LIMIT        = 100   # Deezer max per chart-tracks request
PER_RADIO_LIMIT        = 40    # Radio endpoints return ~40 tracks max
PER_EDITORIAL_LIMIT    = 100
PER_YEAR_LIMIT         = 100

# Year range for the per-year search source.
YEAR_RANGE = (1960, 2025)

DEFAULT_TRACK_CAP   = 10000
DEFAULT_CONCURRENCY = 2

# Discovery sources the script can run. Each name maps to a builder function
# down below; --sources lets the user run subsets.
ALL_SOURCES = ["genres", "editorial", "radio", "years", "playlists", "albums", "cache_expand"]

# Album-walk settings. Top-tracks caps out at ~50 songs per artist; walking
# full discographies reaches every album cut. This is the source that gets
# the pool toward a million: 5k artists x ~15 albums x ~12 tracks ≈ 900k
# candidates before dedupe.
ALBUMS_PER_ARTIST    = 25
ALBUM_ARTIST_CAP     = 5000   # seed artists per run; re-runs pick up new ones

# How many top tracks to pull per artist when expanding from the cache.
EXPAND_TOP_TRACKS_PER_ARTIST       = 50
EXPAND_RELATED_PER_ARTIST          =  8
EXPAND_TOP_TRACKS_PER_RELATED      = 10

# Playlist crawl settings. Playlists are the deepest reservoir on Deezer:
# user- and editor-made lists reach long-tail catalog that charts and radios
# never surface. ~90 keywords x 6 playlists x up to 200 tracks each can
# yield 60k+ candidates before dedupe.
PLAYLISTS_PER_KEYWORD = 6
PLAYLIST_PAGE_SIZE    = 100
PLAYLIST_MAX_PAGES    = 2      # first 200 tracks per playlist

# Keyword bank: genres x moods x decades x activities x regions. Each term
# is a playlist search query, and each finds different corners of the catalog.
PLAYLIST_KEYWORDS = [
    # Core genres
    "pop hits", "rock classics", "indie", "alternative", "hip hop", "rap",
    "r&b", "soul", "funk", "jazz", "blues", "country", "folk", "acoustic",
    "metal", "punk", "grunge", "emo", "ska", "disco", "motown", "gospel",
    # Electronic
    "edm", "house", "deep house", "techno", "trance", "dubstep",
    "drum and bass", "ambient", "synthwave", "lofi", "chillhop",
    # Global
    "latin", "reggaeton", "salsa", "bachata", "afrobeats", "amapiano",
    "k-pop", "j-pop", "bollywood", "arabic pop", "french pop",
    "italian classics", "brazilian", "reggae", "dancehall", "celtic",
    # Decades
    "60s hits", "70s hits", "80s hits", "90s hits", "2000s hits",
    "2010s hits", "oldies",
    # Moods & activities
    "workout", "running", "party", "chill", "study", "focus", "sleep",
    "road trip", "summer vibes", "rainy day", "sad songs", "feel good",
    "romantic", "wedding", "breakup", "dinner party", "cooking", "gaming",
    "morning coffee", "late night", "driving", "beach",
    # Texture / niche
    "classical essentials", "film scores", "instrumental", "piano",
    "guitar", "female vocalists", "one hit wonders", "hidden gems",
    "underrated", "slow jams", "throwback", "covers", "unplugged",
]


# ─── HTTP helper ─────────────────────────────────────────────────────────────

async def _get(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    """Hit a Deezer endpoint, return decoded JSON or None on any failure.

    Handles Deezer's quota responses: the API returns HTTP 200 with
    {"error": {"code": 4}} when the ~50-requests-per-5-seconds limit is
    exceeded. Long crawls (albums/playlists in turbo mode) hit this
    routinely, so back off and retry instead of silently losing the batch.
    """
    for attempt in range(retries):
        try:
            r = await client.http_client.get(url, params=params or {}, timeout=10.0)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            data = r.json()
            if isinstance(data, dict) and (data.get("error") or {}).get("code") == 4:
                await asyncio.sleep(5.0 * (attempt + 1))  # quota — back off
                continue
            return data
        except Exception as e:
            if attempt == retries - 1:
                print(f"  fetch error ({url}): {e}")
            else:
                await asyncio.sleep(1.0 * (attempt + 1))
    return None


# ─── Source 1: Genre charts ──────────────────────────────────────────────────

async def discover_from_genres() -> list[dict]:
    """Walk every Deezer genre and pull its top tracks."""
    print(f"\n[1/4] Discovering from genre charts...")

    data = await _get("https://api.deezer.com/genre")
    genres = (data or {}).get("data", []) or []
    print(f"      Got {len(genres)} genres.")

    # Include the "All" chart too (genre id 0) for general top tracks.
    entries = [{"id": 0, "name": "All"}] + genres

    tracks: list[dict] = []
    for entry in entries:
        gid, gname = entry["id"], entry["name"]
        data = await _get(
            f"https://api.deezer.com/chart/{gid}/tracks",
            params={"limit": PER_GENRE_LIMIT},
        )
        batch = (data or {}).get("data", []) or []
        tracks.extend(batch)
        print(f"      {gname[:22]:22s} (genre {gid:4d}): +{len(batch):3d} tracks")

    return tracks


# ─── Source 2: Editorial sections ────────────────────────────────────────────

async def discover_from_editorial() -> list[dict]:
    """Pull tracks from each editorial section's charts."""
    print(f"\n[2/4] Discovering from editorial sections...")

    data = await _get("https://api.deezer.com/editorial")
    editorials = (data or {}).get("data", []) or []
    print(f"      Got {len(editorials)} editorial sections.")

    tracks: list[dict] = []
    for ed in editorials:
        eid, ename = ed["id"], ed.get("name", "?")
        data = await _get(
            f"https://api.deezer.com/editorial/{eid}/charts/tracks",
            params={"limit": PER_EDITORIAL_LIMIT},
        )
        batch = (data or {}).get("data", []) or []
        tracks.extend(batch)
        print(f"      {ename[:22]:22s} (ed {eid:4d}):    +{len(batch):3d} tracks")

    return tracks


# ─── Source 3: Radio stations ────────────────────────────────────────────────

async def discover_from_radio() -> list[dict]:
    """Pull tracks from every Deezer radio station's seed list."""
    print(f"\n[3/4] Discovering from radio stations...")

    data = await _get("https://api.deezer.com/radio")
    radios = (data or {}).get("data", []) or []
    print(f"      Got {len(radios)} radio stations.")

    tracks: list[dict] = []
    for radio in radios:
        rid, rname = radio["id"], radio.get("title", "?")
        data = await _get(
            f"https://api.deezer.com/radio/{rid}/tracks",
            params={"limit": PER_RADIO_LIMIT},
        )
        batch = (data or {}).get("data", []) or []
        tracks.extend(batch)
        # Radios are numerous; only log every ~10th to keep output readable.
        if (rid % 10) == 0 or len(batch) > 30:
            print(f"      {rname[:22]:22s} (radio {rid:5d}): +{len(batch):3d} tracks")

    return tracks


# ─── Source 4: Per-year searches ─────────────────────────────────────────────

async def discover_from_years() -> list[dict]:
    """Pull top tracks for each year in YEAR_RANGE via the search endpoint."""
    start, end = YEAR_RANGE
    print(f"\n[4/4] Discovering by year ({start}–{end})...")

    tracks: list[dict] = []
    for year in range(start, end + 1):
        data = await _get(
            "https://api.deezer.com/search",
            params={"q": f"year:{year}", "limit": PER_YEAR_LIMIT},
        )
        batch = (data or {}).get("data", []) or []
        tracks.extend(batch)
        if batch:
            print(f"      year {year}: +{len(batch):3d} tracks")

    return tracks


# ─── Source 5: Playlist crawl (deepest catalog reach) ───────────────────────

async def discover_from_playlists() -> list[dict]:
    """Search Deezer playlists across a broad keyword bank and pull their
    tracks.

    Playlists are the single deepest discovery source on Deezer: charts and
    radios surface a few hundred popular tracks per genre, but user- and
    editor-curated playlists dig into album cuts, deep catalog, regional
    scenes, and niche moods. Every keyword lands on different playlists,
    and every playlist holds up to several hundred tracks.
    """
    print(f"\n[+] Discovering from playlists ({len(PLAYLIST_KEYWORDS)} keywords)...")

    tracks: list[dict] = []
    seen_playlists: set[str] = set()

    for kw in PLAYLIST_KEYWORDS:
        data = await _get(
            "https://api.deezer.com/search/playlist",
            params={"q": kw, "limit": PLAYLISTS_PER_KEYWORD},
        )
        playlists = (data or {}).get("data", []) or []

        kw_added = 0
        for pl in playlists:
            pid = str(pl.get("id", ""))
            if not pid or pid in seen_playlists:
                continue
            seen_playlists.add(pid)

            for page in range(PLAYLIST_MAX_PAGES):
                pdata = await _get(
                    f"https://api.deezer.com/playlist/{pid}/tracks",
                    params={
                        "limit": PLAYLIST_PAGE_SIZE,
                        "index": page * PLAYLIST_PAGE_SIZE,
                    },
                )
                batch = (pdata or {}).get("data", []) or []
                tracks.extend(batch)
                kw_added += len(batch)
                if len(batch) < PLAYLIST_PAGE_SIZE:
                    break  # playlist exhausted

        print(
            f"      {kw[:24]:24s}: +{kw_added:4d} tracks "
            f"(playlists crawled: {len(seen_playlists)}, total: {len(tracks)})"
        )

    print(f"\n      → Total playlist candidates: {len(tracks)} (before dedup)")
    return tracks


# ─── Source 6: Album walks (full discographies — the road to 1M) ────────────

async def discover_from_albums() -> list[dict]:
    """Walk the full discography of every cached artist.

    Charts, radios, and top-tracks all skim an artist's most popular songs;
    the albums endpoint reaches everything else — album cuts, B-sides,
    early records. Seeded from cached artists, so every seed is verified to
    have working previews.
    """
    print(f"\n[+] Discovering from album walks...")

    cached_entries = await cache.get_all_cached()
    seed_artists: dict[str, str] = {}
    for entry in cached_entries:
        aid   = entry.get("artist_id")
        aname = entry.get("artist_name") or ""
        # iTunes-space artists ("it:...") aren't walkable on Deezer.
        if aid and aid != "None" and not aid.startswith("it:") and aid not in seed_artists:
            seed_artists[aid] = aname
        if len(seed_artists) >= ALBUM_ARTIST_CAP:
            break
    print(f"      → {len(seed_artists)} seed artists (cap {ALBUM_ARTIST_CAP}).")

    tracks: list[dict] = []
    for i, (aid, aname) in enumerate(seed_artists.items()):
        adata = await _get(
            f"https://api.deezer.com/artist/{aid}/albums",
            params={"limit": ALBUMS_PER_ARTIST},
        )
        albums = (adata or {}).get("data", []) or []

        artist_added = 0
        for album in albums:
            alb_id = album.get("id")
            if not alb_id:
                continue
            cover = album.get("cover_medium", "") or album.get("cover_big", "")
            tdata = await _get(f"https://api.deezer.com/album/{alb_id}/tracks",
                               params={"limit": 100})
            batch = (tdata or {}).get("data", []) or []
            # Album-track objects omit the album payload; splice the cover
            # back in so the cache stores artwork for recommendations.
            for t in batch:
                t.setdefault("album", {"cover_medium": cover})
            tracks.extend(batch)
            artist_added += len(batch)

        if (i + 1) % 25 == 0 or (i + 1) == len(seed_artists):
            print(f"      [{i + 1:4d}/{len(seed_artists)}] {aname[:26]:26s} "
                  f"+{artist_added:4d} tracks ({len(albums)} albums, total: {len(tracks)})")

    print(f"\n      → Total album candidates: {len(tracks)} (before dedup)")
    return tracks


# ─── Source 7: Cache-seeded expansion (smart, high success rate) ────────────

async def discover_from_cache_artists() -> list[dict]:
    """Expand outward from artists we've already successfully cached.

    This is the highest-success-rate discovery strategy because every seed
    artist is verified to have at least one working preview. We exploit
    this to dig deeper into:

      Phase A — Each cached artist's full top-tracks catalog (50 tracks
                each, vs. the 5 we pull during initial discovery)
      Phase B — Related artists of each cached artist, then their top
                tracks (collaborative-filter expansion seeded from
                known-good artists)

    Expected yield is much higher than the broad-source walks because
    artists with one working preview almost always have multiple.
    """
    print(f"\n[+] Discovering from cached artists (smart expansion)...")

    cached_entries = await cache.get_all_cached()
    print(f"      Loaded {len(cached_entries)} entries from cache.")

    # Build a unique-artist set from the cache.
    seed_artists: dict[str, str] = {}  # artist_id -> artist_name
    for entry in cached_entries:
        aid   = entry.get("artist_id")
        aname = entry.get("artist_name") or ""
        if aid and aid != "None" and aid not in seed_artists:
            seed_artists[aid] = aname
    print(f"      → {len(seed_artists)} unique seed artists.")

    tracks: list[dict] = []

    # ── Phase A: Deeper top tracks per cached artist ─────────────────
    print(f"\n      [A] Fetching deeper top tracks per seed artist...")
    for i, (aid, aname) in enumerate(seed_artists.items()):
        data = await _get(
            f"https://api.deezer.com/artist/{aid}/top",
            params={"limit": EXPAND_TOP_TRACKS_PER_ARTIST},
        )
        batch = (data or {}).get("data", []) or []
        tracks.extend(batch)
        # Log progress every 25 artists so output stays scannable.
        if (i + 1) % 25 == 0 or (i + 1) == len(seed_artists):
            print(f"          [{i + 1:4d}/{len(seed_artists)}] {aname[:28]:28s} +{len(batch):3d} tracks (total: {len(tracks)})")

    # ── Phase B: Related artists' top tracks ─────────────────────────
    print(f"\n      [B] Fetching related artists, then their top tracks...")
    related_seen: set[str] = set()

    for i, (aid, aname) in enumerate(seed_artists.items()):
        data = await _get(
            f"https://api.deezer.com/artist/{aid}/related",
            params={"limit": EXPAND_RELATED_PER_ARTIST},
        )
        related = (data or {}).get("data", []) or []

        new_related_count = 0
        for rel in related:
            rel_id = str(rel.get("id", ""))
            if not rel_id or rel_id in related_seen or rel_id in seed_artists:
                continue
            related_seen.add(rel_id)
            new_related_count += 1

            # Fetch top tracks for this related artist.
            rt_data = await _get(
                f"https://api.deezer.com/artist/{rel_id}/top",
                params={"limit": EXPAND_TOP_TRACKS_PER_RELATED},
            )
            rt_batch = (rt_data or {}).get("data", []) or []
            tracks.extend(rt_batch)

        if (i + 1) % 25 == 0 or (i + 1) == len(seed_artists):
            print(
                f"          [{i + 1:4d}/{len(seed_artists)}] {aname[:28]:28s} "
                f"+{new_related_count} new related (related pool: {len(related_seen)}, "
                f"total tracks: {len(tracks)})"
            )

    print(f"\n      → Total candidates collected: {len(tracks)} (before dedup)")
    return tracks


# ─── Discovery orchestrator ──────────────────────────────────────────────────

SOURCE_FUNCS = {
    "genres":       discover_from_genres,
    "editorial":    discover_from_editorial,
    "radio":        discover_from_radio,
    "years":        discover_from_years,
    "playlists":    discover_from_playlists,
    "albums":       discover_from_albums,
    "cache_expand": discover_from_cache_artists,
}


def deezer_track_to_dict(t: dict) -> dict:
    artist = t.get("artist", {}) or {}
    album  = t.get("album",  {}) or {}
    return {
        "name":      t.get("title", ""),
        "artist":    artist.get("name", ""),
        "artist_id": str(artist.get("id", "")),
        "track_id":  str(t.get("id", "")),
        "deezer_id": t.get("id"),
        "image":     album.get("cover_medium", "") or album.get("cover_big", ""),
        "url":       t.get("link", ""),
        "preview":   t.get("preview", ""),
    }


async def discover_candidates(sources: list[str], target: int) -> list[dict]:
    """Run each requested source, dedup by track ID, cap at target."""
    candidates: dict[str, dict] = {}

    for source_name in sources:
        if source_name not in SOURCE_FUNCS:
            print(f"  ! unknown source: {source_name} — skipping")
            continue
        if len(candidates) >= target:
            print(f"  Reached target of {target} unique tracks, stopping.")
            break

        raw_tracks = await SOURCE_FUNCS[source_name]()

        added = 0
        for t in raw_tracks:
            tid = str(t.get("id", ""))
            if tid and tid not in candidates:
                candidates[tid] = deezer_track_to_dict(t)
                added += 1
        print(f"      → {source_name}: +{added} unique  (running pool: {len(candidates)})")

    return list(candidates.values())[:target]


# ─── The actual analysis loop ───────────────────────────────────────────────

# Chunk size for the "already cached?" pre-check. One ANY($1) query per
# chunk keeps both the query planner and client memory happy at 1M IDs.
EXISTING_CHECK_CHUNK = 5000

# Print a rate/ETA summary line every N analyzed tracks.
PROGRESS_EVERY = 200


async def prewarm(sources: list[str], target: int, concurrency: int) -> None:
    candidates = await discover_candidates(sources=sources, target=target)
    print(f"\n{'='*60}")
    print(f"Discovered {len(candidates)} unique candidate tracks across "
          f"{len(sources)} sources.")

    if not candidates:
        return

    # Skip anything already in the cache (chunked, existence-check only —
    # this is what makes re-runs resumable: interrupt any time, run again,
    # and it picks up where it left off).
    ids = [str(c["deezer_id"]) for c in candidates]
    existing: set[str] = set()
    for j in range(0, len(ids), EXISTING_CHECK_CHUNK):
        existing |= await cache.get_cached_ids(ids[j:j + EXISTING_CHECK_CHUNK])
    fresh = [c for c in candidates if str(c["deezer_id"]) not in existing]
    print(f"Already cached: {len(existing)}")
    print(f"To analyze:     {len(fresh)}\n")

    if not fresh:
        print("Nothing to do — the cache is fully warmed for this pool.")
        return

    # deezer.enrich_track serializes analyses through a Semaphore(1) sized
    # for the memory-tight production Space. Prewarm runs on a real machine,
    # so widen the gate to the requested concurrency — without this the
    # --concurrency flag only parallelizes downloads, not analysis.
    deezer._analysis_semaphore = asyncio.Semaphore(concurrency)

    completed = 0
    failed    = 0
    start     = time.time()

    async def warm_one(track: dict, idx: int):
        nonlocal completed, failed
        t0 = time.time()
        try:
            result = await deezer.enrich_track(track)
            if result.get("vector") is None:
                failed += 1
                status = "FAIL"
            else:
                completed += 1
                status = "OK  "
        except Exception as e:
            failed += 1
            status = f"ERR ({type(e).__name__})"
        elapsed = time.time() - t0
        print(
            f"  [{idx + 1:5d}/{len(fresh)}] {status} "
            f"{elapsed:5.2f}s — {track['artist'][:25]:25s} — {track['name'][:50]}"
        )

        done = completed + failed
        if done % PROGRESS_EVERY == 0:
            rate = done / max(time.time() - start, 1e-9)
            remaining = len(fresh) - done
            eta_h = remaining / max(rate, 1e-9) / 3600
            print(
                f"  {'─'*56}\n"
                f"  progress: {done}/{len(fresh)} "
                f"({completed} ok, {failed} failed) — "
                f"{rate:.2f} trk/s, ETA {eta_h:.1f}h\n"
                f"  {'─'*56}"
            )

    # Worker pool: N workers pull from a shared cursor. Unlike spawning one
    # task per track, this stays memory-flat at any pool size.
    cursor = {"next": 0}

    async def worker():
        while True:
            i = cursor["next"]
            if i >= len(fresh):
                return
            cursor["next"] = i + 1
            await warm_one(fresh[i], i)

    await asyncio.gather(*[worker() for _ in range(concurrency)])

    total_elapsed = time.time() - start
    final_count   = await cache.count_cached()
    print(f"\n{'='*60}")
    print(f"Done in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Newly analyzed: {completed}")
    print(f"  Failed:         {failed}")
    print(f"  Cache total:    {final_count}")
    if completed > 0:
        print(f"  Avg time/track: {total_elapsed / completed:.2f}s")


# Turbo defaults: the deep sources, real parallelism, and a cap sized for
# overnight runs. Explicit flags always win over these.
TURBO_TRACK_CAP   = 200_000
TURBO_CONCURRENCY = 8
TURBO_SOURCES     = "playlists,albums,cache_expand"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks", type=int, default=None,
        help=f"Cap on unique tracks to pre-warm. Default {DEFAULT_TRACK_CAP} "
             f"(turbo: {TURBO_TRACK_CAP}).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help=(
            f"Parallel librosa analyses. Laptop can handle 4-8; "
            f"leave at {DEFAULT_CONCURRENCY} for shared/free hosting "
            f"(turbo: {TURBO_CONCURRENCY})."
        ),
    )
    parser.add_argument(
        "--sources", type=str, default=None,
        help=(
            f"Comma-separated discovery sources. Available: {','.join(ALL_SOURCES)}. "
            f"Default runs all of them in order (turbo: {TURBO_SOURCES})."
        ),
    )
    parser.add_argument(
        "--turbo", action="store_true",
        help=(
            "Deep-crawl mode for growing the pool toward 1M: playlist + "
            "album-discography + artist-graph sources, 8-way analysis, "
            "200k track cap. Resumable — interrupt and re-run freely."
        ),
    )
    args = parser.parse_args()

    if args.turbo:
        tracks      = args.tracks      or TURBO_TRACK_CAP
        concurrency = args.concurrency or TURBO_CONCURRENCY
        sources_arg = args.sources     or TURBO_SOURCES
    else:
        tracks      = args.tracks      or DEFAULT_TRACK_CAP
        concurrency = args.concurrency or DEFAULT_CONCURRENCY
        sources_arg = args.sources     or ",".join(ALL_SOURCES)

    sources = [s.strip() for s in sources_arg.split(",") if s.strip()]

    client.http_client = httpx.AsyncClient(timeout=30.0)
    await cache.init_pool()
    try:
        await prewarm(
            sources=sources,
            target=tracks,
            concurrency=concurrency,
        )
    finally:
        await client.http_client.aclose()
        await cache.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
