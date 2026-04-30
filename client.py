import httpx

# Global HTTP client to be shared across modules.
# It is initialized and closed in the FastAPI lifespan in app_v2.py.
http_client: httpx.AsyncClient = None
