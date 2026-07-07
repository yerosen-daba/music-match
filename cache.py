"""
cache.py — Persistent feature cache backed by Supabase Postgres.

Replaces the old in-memory _feature_cache dict in deezer.py (which died on
every server restart) with a real database. Once a song's audio features
have been computed, they're stored here forever and served sub-100ms on
every subsequent request — across users, across restarts, across deploys.

Schema (created on app startup if missing):

    feature_cache
      deezer_id    TEXT       PRIMARY KEY    -- Deezer track ID
      features     JSONB      NOT NULL       -- full feature dict
      vector       JSONB      NOT NULL       -- 53-dim feature vector
      track_name   TEXT                      -- for debugging / browsing
      artist_name  TEXT                      -- for debugging / browsing
      created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
      updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()

Connection details come from the DATABASE_URL environment variable, which
must be the Supabase Session Pooler URI (IPv4-compatible, port 5432).
"""

import json
import os

import asyncpg


# ─── Module state ────────────────────────────────────────────────────────────

# Connection pool, initialized via init_pool() during app startup.
_pool: asyncpg.Pool | None = None

# True when the pgvector extension + HNSW index are live. When False,
# match.py falls back to the legacy in-memory matrix recommender.
_pgvector: bool = False


# ─── Schema migration ────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feature_cache (
    deezer_id    TEXT PRIMARY KEY,
    features     JSONB NOT NULL,
    vector       JSONB NOT NULL,
    track_name   TEXT,
    artist_name  TEXT,
    artist_id    TEXT,
    track_url    TEXT,
    track_image  TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Idempotent column adds, in case this table was created by an earlier
-- version of the schema. Safe to leave in forever.
ALTER TABLE feature_cache ADD COLUMN IF NOT EXISTS artist_id   TEXT;
ALTER TABLE feature_cache ADD COLUMN IF NOT EXISTS track_url   TEXT;
ALTER TABLE feature_cache ADD COLUMN IF NOT EXISTS track_image TEXT;

CREATE INDEX IF NOT EXISTS idx_feature_cache_artist
  ON feature_cache (artist_name);
"""


# ─── Lifecycle ───────────────────────────────────────────────────────────────

async def init_pool(dsn: str | None = None) -> None:
    """Create the connection pool and apply the schema.

    Idempotent. Call once from the FastAPI lifespan on startup.

    Args:
        dsn: Postgres connection string. If not provided, reads from the
             DATABASE_URL environment variable.
    """
    global _pool

    if _pool is not None:
        return  # already initialized

    dsn = dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL environment variable not set. "
            "Expected the Supabase Session Pooler URI."
        )

    # Supabase requires TLS. Pass ssl='require' so asyncpg negotiates it.
    # min_size kept low so the app starts fast even on cold boot;
    # max_size kept modest because Supabase's free tier has connection limits.
    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=5,
        ssl="require",
        command_timeout=10.0,
        # Disable prepared statement caching to play nicely with poolers.
        statement_cache_size=0,
    )

    # Apply schema. Safe to re-run on every startup.
    global _pgvector
    async with _pool.acquire() as conn:
        await conn.execute(_SCHEMA_SQL, timeout=60)
        _pgvector = await _init_pgvector(conn)


async def _init_pgvector(conn) -> bool:
    """Set up the pgvector column + HNSW index. Returns True on success.

    All statements are idempotent and additive (nothing is dropped), so this
    runs safely on every startup. The backfill converts existing JSONB
    vectors into the native vector column — a JSONB array's text form
    ("[0.1, 0.2, ...]") is exactly pgvector's input format, so no Python
    round-trip is needed.

    Generous timeouts: on a large table the first backfill + HNSW build can
    take minutes. Subsequent startups are no-ops.
    """
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector", timeout=30)
    except Exception as e:
        print(f"[cache] pgvector extension unavailable ({e}); "
              f"recommendations will use the in-memory matrix fallback")
        return False
    try:
        await conn.execute(
            "ALTER TABLE feature_cache ADD COLUMN IF NOT EXISTS embedding vector(53)",
            timeout=30,
        )
        await conn.execute(
            "UPDATE feature_cache SET embedding = (vector::text)::vector "
            "WHERE embedding IS NULL",
            timeout=600,
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feature_cache_embedding "
            "ON feature_cache USING hnsw (embedding vector_cosine_ops)",
            timeout=600,
        )
        return True
    except Exception as e:
        print(f"[cache] pgvector setup failed ({e}); "
              f"recommendations will use the in-memory matrix fallback")
        return False


def pgvector_enabled() -> bool:
    """Whether nearest-neighbor search can run inside Postgres."""
    return _pgvector


async def close_pool() -> None:
    """Tear down the pool. Call from FastAPI lifespan shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ─── Read paths ──────────────────────────────────────────────────────────────

async def get_cached(deezer_id: str | int) -> dict | None:
    """Look up cached features for one track. Returns the analyzer-shape dict
    {"features": ..., "vector": [...]} or None if not in cache.
    """
    if _pool is None or deezer_id is None:
        return None

    row = await _pool.fetchrow(
        "SELECT features, vector FROM feature_cache WHERE deezer_id = $1",
        str(deezer_id),
    )
    if row is None:
        return None

    return {
        "features": json.loads(row["features"]) if isinstance(row["features"], str) else row["features"],
        "vector":   json.loads(row["vector"])   if isinstance(row["vector"], str)   else row["vector"],
    }


async def get_cached_ids(deezer_ids: list[str | int]) -> set[str]:
    """Which of these IDs are already cached? Existence check only — no
    JSONB parsing, so it stays cheap even for very large batches (prewarm
    calls this in chunks while planning a crawl).
    """
    if _pool is None or not deezer_ids:
        return set()
    str_ids = [str(d) for d in deezer_ids if d is not None]
    if not str_ids:
        return set()
    rows = await _pool.fetch(
        "SELECT deezer_id FROM feature_cache WHERE deezer_id = ANY($1::text[])",
        str_ids,
    )
    return {row["deezer_id"] for row in rows}


async def get_many_cached(deezer_ids: list[str | int]) -> dict[str, dict]:
    """Batch lookup. Returns a dict mapping deezer_id (string) → cached entry.
    Missing IDs are simply not present in the returned dict.

    This is the hot path for the recommendation engine: pull 100 candidate
    track IDs, hit the DB once, get back whatever's already analyzed.
    """
    if _pool is None or not deezer_ids:
        return {}

    str_ids = [str(d) for d in deezer_ids if d is not None]
    if not str_ids:
        return {}

    rows = await _pool.fetch(
        "SELECT deezer_id, features, vector FROM feature_cache "
        "WHERE deezer_id = ANY($1::text[])",
        str_ids,
    )

    result: dict[str, dict] = {}
    for row in rows:
        result[row["deezer_id"]] = {
            "features": json.loads(row["features"]) if isinstance(row["features"], str) else row["features"],
            "vector":   json.loads(row["vector"])   if isinstance(row["vector"], str)   else row["vector"],
        }
    return result


# ─── Write path ──────────────────────────────────────────────────────────────

async def set_cached(
    deezer_id: str | int,
    features: dict,
    vector: list[float],
    track_name: str | None = None,
    artist_name: str | None = None,
    artist_id: str | None = None,
    track_url: str | None = None,
    track_image: str | None = None,
) -> None:
    """Upsert a track's analysis into the cache.

    Stores the audio features + vector alongside enough metadata that the
    midpoint-search recommender can return tracks straight out of the cache
    without having to round-trip to Deezer for image/URL/artist info.

    Uses ON CONFLICT so re-analyzing a track (e.g. after we improve the
    feature pipeline) overwrites the old entry. updated_at gets bumped.
    """
    if _pool is None or deezer_id is None:
        return

    # A JSON array literal is also valid pgvector input, so $3 feeds both
    # the legacy JSONB column and the native embedding column.
    embedding_col   = ", embedding"                 if _pgvector else ""
    embedding_value = ", $3::vector"                if _pgvector else ""
    embedding_set   = ", embedding = EXCLUDED.embedding" if _pgvector else ""

    await _pool.execute(
        f"""
        INSERT INTO feature_cache
            (deezer_id, features, vector, track_name, artist_name,
             artist_id, track_url, track_image{embedding_col})
        VALUES ($1, $2::jsonb, $3::jsonb, $4, $5, $6, $7, $8{embedding_value})
        ON CONFLICT (deezer_id) DO UPDATE
            SET features    = EXCLUDED.features,
                vector      = EXCLUDED.vector,
                track_name  = COALESCE(EXCLUDED.track_name,  feature_cache.track_name),
                artist_name = COALESCE(EXCLUDED.artist_name, feature_cache.artist_name),
                artist_id   = COALESCE(EXCLUDED.artist_id,   feature_cache.artist_id),
                track_url   = COALESCE(EXCLUDED.track_url,   feature_cache.track_url),
                track_image = COALESCE(EXCLUDED.track_image, feature_cache.track_image),
                updated_at  = NOW(){embedding_set}
        """,
        str(deezer_id),
        json.dumps(features),
        json.dumps(vector),
        track_name,
        artist_name,
        artist_id,
        track_url,
        track_image,
    )


async def get_all_cached() -> list[dict]:
    """Return every cached track's vector + metadata.

    Used by the midpoint-search recommender. The expectation is that callers
    cache the result in-process for the lifetime of a few requests rather
    than calling this every time — at scale this pulls all rows from the
    table, which is a lot of bytes over the wire.

    Each returned dict has:
        deezer_id, vector (list[float]), track_name, artist_name,
        artist_id, track_url, track_image
    """
    if _pool is None:
        return []

    rows = await _pool.fetch(
        "SELECT deezer_id, vector, track_name, artist_name, "
        "       artist_id, track_url, track_image "
        "FROM feature_cache"
    )

    result: list[dict] = []
    for row in rows:
        vec = row["vector"]
        if isinstance(vec, str):
            vec = json.loads(vec)
        result.append({
            "deezer_id":   row["deezer_id"],
            "vector":      vec,
            "track_name":  row["track_name"],
            "artist_name": row["artist_name"],
            "artist_id":   row["artist_id"],
            "track_url":   row["track_url"],
            "track_image": row["track_image"],
        })
    return result


# ─── Nearest-neighbor search (pgvector) ──────────────────────────────────────

async def nearest_neighbors(vector: list[float], limit: int = 120) -> list[dict]:
    """Top-N cached tracks by cosine similarity to the given vector, computed
    inside Postgres via the HNSW index.

    This replaces pulling the whole table into an in-memory numpy matrix —
    it stays fast and memory-flat no matter how large the cache grows.

    Returns dicts with metadata + "cosine_sim", sorted most-similar first.
    Empty list if pgvector isn't available (callers fall back to the matrix).
    """
    if _pool is None or not _pgvector:
        return []

    vec_txt = json.dumps(vector)
    async with _pool.acquire() as conn:
        async with conn.transaction():
            # HNSW's default candidate-list size (ef_search=40) is smaller
            # than our overfetch; raise it for this query only.
            await conn.execute("SET LOCAL hnsw.ef_search = 200")
            rows = await conn.fetch(
                """
                SELECT deezer_id, track_name, artist_name, artist_id,
                       track_url, track_image,
                       1 - (embedding <=> $1::vector) AS cosine_sim
                FROM feature_cache
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_txt,
                limit,
            )
    return [dict(r) for r in rows]


# ─── Observability ───────────────────────────────────────────────────────────

async def count_cached() -> int:
    """Return total number of cached tracks. Useful for health checks and
    monitoring the pre-warm progress."""
    if _pool is None:
        return 0
    row = await _pool.fetchrow("SELECT COUNT(*) AS n FROM feature_cache")
    return int(row["n"]) if row else 0
