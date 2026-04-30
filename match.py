import math
import asyncio
import client
from deezer import enrich_track

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def gaussian_sim(diff: float, sigma: float) -> float:
    """Gaussian similarity: 1.0 when diff=0, drops toward 0 as diff grows."""
    return math.exp(-(diff * diff) / (2 * sigma * sigma))

def compute_compatibility(songs1: list[dict], songs2: list[dict]) -> dict:
    """
    Original 0–100 score using Gaussian similarity on audio features.

    Breakdown:
      Energy  — similar loudness/intensity profiles        up to 40 pts
      Tempo   — similar BPM preferences                    up to 30 pts
      Mood    — similar valence (happy/sad feel)            up to 30 pts
    """
    # Average features for each person
    e1 = mean([s["energy"]  for s in songs1])
    e2 = mean([s["energy"]  for s in songs2])
    t1 = mean([s["tempo"]   for s in songs1])
    t2 = mean([s["tempo"]   for s in songs2])
    v1 = mean([s["valence"] for s in songs1])
    v2 = mean([s["valence"] for s in songs2])

    # Gaussian similarity — ultra-strict, only true matches score high
    energy_sim  = gaussian_sim(abs(e1 - e2), 0.04)
    tempo_sim   = gaussian_sim(abs(t1 - t2), 4.0)     # 4 BPM sigma
    valence_sim = gaussian_sim(abs(v1 - v2), 0.04)

    energy_score  = round(energy_sim  * 40)
    tempo_score   = round(tempo_sim   * 30)
    valence_score = round(valence_sim * 30)

    total = energy_score + tempo_score + valence_score

    # ── Vibe labels based on feature averages ──
    def vibe_label(energy, tempo, valence):
        labels = []
        if energy > 0.65:
            labels.append("High Energy")
        elif energy < 0.35:
            labels.append("Chill")
        else:
            labels.append("Moderate")

        if valence > 0.6:
            labels.append("Upbeat")
        elif valence < 0.4:
            labels.append("Moody")
        else:
            labels.append("Balanced")

        return " / ".join(labels)

    return {
        "total":         min(total, 100),
        "energy_score":  energy_score,
        "tempo_score":   tempo_score,
        "valence_score": valence_score,
        "vibe1":         vibe_label(e1, t1, v1),
        "vibe2":         vibe_label(e2, t2, v2),
        "details": {
            "avg_energy_1":  round(e1, 3),
            "avg_energy_2":  round(e2, 3),
            "avg_tempo_1":   round(t1, 1),
            "avg_tempo_2":   round(t2, 1),
            "avg_valence_1": round(v1, 3),
            "avg_valence_2": round(v2, 3),
        },
    }

def compatibility_message(score: int) -> str:
    if score >= 85: return "Soulmates 🎵 You were made for the same playlist."
    if score >= 70: return "Great match! You'd have an amazing road trip together."
    if score >= 55: return "Solid overlap. You'd agree on at least half the aux cord."
    if score >= 40: return "Some common ground. You'd negotiate the playlist."
    if score >= 25: return "Different worlds. But opposites attract, right?"
    return "Complete opposites. Take turns on the aux — no fighting."

async def get_recommendations(songs1: list[dict], songs2: list[dict]) -> list[dict]:
    """
    Find 6 NEW song recommendations matched by audio characteristics.

    For each person:
      1. Compute their average audio profile (energy, tempo, valence)
      2. Discover related artists via Deezer
      3. Get candidate tracks and enrich them with librosa
      4. Rank candidates by how close their features are to the person's average
      5. Pick the top 3 closest matches

    Returns 6 songs, alternating: P1, P2, P1, P2, P1, P2
    """
    # ── Step 1: Compute average audio profiles from already-enriched songs ──
    avg1 = {
        "energy":  mean([s["energy"]  for s in songs1]),
        "tempo":   mean([s["tempo"]   for s in songs1]),
        "valence": mean([s["valence"] for s in songs1]),
    }
    avg2 = {
        "energy":  mean([s["energy"]  for s in songs2]),
        "tempo":   mean([s["tempo"]   for s in songs2]),
        "valence": mean([s["valence"] for s in songs2]),
    }

    # IDs to exclude: input songs + input artists
    existing_track_ids = {s["track_id"] for s in songs1 + songs2}
    existing_artist_ids = {s.get("artist_id", "") for s in songs1 + songs2}

    def unique_artist_ids(songs):
        seen, result = set(), []
        for s in songs:
            aid = s.get("artist_id", "")
            if aid and aid not in seen:
                seen.add(aid)
                result.append(aid)
        return result

    artist_ids1 = unique_artist_ids(songs1)
    artist_ids2 = unique_artist_ids(songs2)

    # ── Step 2: Discover related artists via Deezer ──
    async def fetch_related(artist_id: str):
        try:
            r = await client.http_client.get(
                f"https://api.deezer.com/artist/{artist_id}/related",
                params={"limit": 5}
            )
            if r.status_code == 200:
                return r.json().get("data", [])
        except Exception:
            pass
        return []

    all_ids = artist_ids1 + artist_ids2
    related_batches = await asyncio.gather(*[
        fetch_related(aid) for aid in all_ids
    ])

    # Build pools of NEW related artists (exclude ones already in input)
    def build_new_pool(batches):
        seen, pool = set(), []
        for batch in batches:
            for a in batch:
                aid = str(a.get("id", ""))
                if aid and aid not in existing_artist_ids and aid not in seen:
                    seen.add(aid)
                    pool.append(aid)
        return pool

    new_artist_ids1 = build_new_pool(related_batches[:len(artist_ids1)])[:6]
    new_artist_ids2 = build_new_pool(related_batches[len(artist_ids1):])[:6]

    # ── Step 3: Get top tracks from related artists ──
    async def fetch_top_tracks(artist_id: str, limit: int = 3):
        try:
            r = await client.http_client.get(
                f"https://api.deezer.com/artist/{artist_id}/top",
                params={"limit": limit}
            )
            if r.status_code == 200:
                return r.json().get("data", [])
        except Exception:
            pass
        return []

    all_new_ids = new_artist_ids1 + new_artist_ids2
    track_batches = await asyncio.gather(*[
        fetch_top_tracks(aid, limit=1) for aid in all_new_ids
    ])

    # Build candidate track dicts (formatted for enrich_track)
    def build_candidates(batches):
        seen, candidates = set(), []
        for batch in batches:
            for t in batch:
                tid = str(t.get("id", ""))
                if tid and tid not in existing_track_ids and tid not in seen:
                    seen.add(tid)
                    artist = t.get("artist", {})
                    album = t.get("album", {})
                    candidates.append({
                        "name":      t.get("title", ""),
                        "artist":    artist.get("name", ""),
                        "artist_id": str(artist.get("id", "")),
                        "track_id":  tid,
                        "deezer_id": t.get("id"),
                        "image":     album.get("cover_medium", "") or album.get("cover_big", ""),
                        "url":       t.get("link", ""),
                        "preview":   t.get("preview", ""),
                    })
        return candidates

    candidates1 = build_candidates(track_batches[:len(new_artist_ids1)])[:6]
    candidates2 = build_candidates(track_batches[len(new_artist_ids1):])[:6]

    # ── Step 4: Enrich candidates with librosa (energy, tempo, valence) ──
    all_candidates = candidates1 + candidates2
    enriched_all = await asyncio.gather(*[
        enrich_track(c) for c in all_candidates
    ])
    enriched1 = enriched_all[:len(candidates1)]
    enriched2 = enriched_all[len(candidates1):]

    # ── Step 5: Rank by feature similarity to the person's average ──
    def feature_distance(track, avg):
        """Weighted distance: energy and valence on 0-1 scale, tempo normalized."""
        e_diff = abs(track.get("energy", 0.5) - avg["energy"])
        t_diff = abs(track.get("tempo", 120) - avg["tempo"]) / 200.0
        v_diff = abs(track.get("valence", 0.5) - avg["valence"])
        # Weight: energy 35%, tempo 35%, valence 30%
        return e_diff * 0.35 + t_diff * 0.35 + v_diff * 0.30

    scored1 = sorted(enriched1, key=lambda t: feature_distance(t, avg1))
    scored2 = sorted(enriched2, key=lambda t: feature_distance(t, avg2))

    # Pick top 3 per person (unique artists only)
    def pick_top(scored, n=3):
        result, used_artists = [], set()
        for t in scored:
            aid = t.get("artist_id", "")
            if aid in used_artists:
                continue
            result.append({
                "name":   t["name"],
                "artist": t["artist"],
                "image":  t.get("image", ""),
                "url":    t.get("url", ""),
            })
            used_artists.add(aid)
            if len(result) >= n:
                break
        return result

    top1 = pick_top(scored1, 3)
    top2 = pick_top(scored2, 3)

    # ── Step 6: Interleave P1, P2, P1, P2, P1, P2 ──
    final = []
    for i in range(3):
        if i < len(top1):
            final.append(top1[i])
        if i < len(top2):
            final.append(top2[i])

    return final
