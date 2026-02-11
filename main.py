"""
Doc-Squeeze API — Knowledge extraction API for AI agents.
Stack: FastAPI + Jina Reader (scraping) + Groq/Llama3 (intelligence).

Capabilities:
- /api/squeeze   → Fetch URL as clean markdown (with optional focus filter)
- /api/extract   → Structured JSON extraction from any URL + schema
- /api/batch     → Parallel multi-URL processing
- /api/search    → Deep search within a document
- Smart caching, API key auth, MCP + OpenAI plugin discovery
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cache import SmartCache, content_cache

BASE_DIR = Path(__file__).resolve().parent

# ── Bootstrap ────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
)
logger = logging.getLogger("doc-squeeze")


# ── Configuration ────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    """Centralized configuration using Pydantic BaseSettings."""

    # API Keys
    groq_api_key: Optional[str] = None

    # Server
    port: int = 8000
    env: str = "development"

    # CORS - comma-separated list of allowed origins
    allowed_origins: str = "*"

    # Timeouts (seconds)
    jina_timeout: int = 30
    groq_timeout: int = 45

    # Content Processing
    max_raw_chars: int = 20_000  # Threshold for triggering Groq filtering
    groq_max_input_chars: int = 60_000  # ~15k tokens with 4 chars/token estimate
    groq_max_output_tokens: int = 8192

    # External APIs
    jina_prefix: str = "https://r.jina.ai/"
    groq_model: str = "llama-3.3-70b-versatile"

    # Rate Limiting
    rate_limit_squeeze: str = "10/minute"  # Format: "requests/time_unit"
    rate_limit_free: str = "5/hour"  # Unauthenticated requests

    # Cache
    cache_ttl: int = 900  # 15 minutes
    cache_max_entries: int = 500

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    @property
    def is_development(self) -> bool:
        return self.env.lower() in ("development", "dev", "local")

    @property
    def cors_origins(self) -> list[str]:
        """Parse comma-separated origins."""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]


settings = Settings()


# ── Security: Blocked hosts for URL validation ───────────────────────────────
BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0",
    "169.254.169.254",  # AWS metadata endpoint
    "[::1]",  # IPv6 localhost
}
BLOCKED_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.")


def _validate_url(v: HttpUrl) -> HttpUrl:
    """Shared URL security validation."""
    parsed = urlparse(str(v))
    if parsed.scheme not in ["http", "https"]:
        raise ValueError("Only HTTP and HTTPS URLs are allowed")
    hostname = parsed.hostname or ""
    if hostname.lower() in BLOCKED_HOSTS:
        raise ValueError("Internal/localhost URLs are not allowed")
    for prefix in BLOCKED_PREFIXES:
        if hostname.startswith(prefix):
            raise ValueError("Private IP addresses are not allowed")
    return v


# ── Models ───────────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: str
    detail: str
    code: str


class SqueezeRequest(BaseModel):
    """Request model with URL validation."""
    url: HttpUrl
    focus: Optional[str] = None

    @field_validator('url')
    @classmethod
    def validate_url_security(cls, v: HttpUrl) -> HttpUrl:
        return _validate_url(v)


class TimingInfo(BaseModel):
    """Request timing breakdown."""
    fetch_ms: int
    filter_ms: Optional[int] = None
    extract_ms: Optional[int] = None
    total_ms: int


class SqueezeResponse(BaseModel):
    """Response model for squeeze endpoint."""
    status: str
    markdown: str
    source: str
    char_count: int
    raw_char_count: int
    was_filtered: bool
    model_used: Optional[str] = None
    timing: TimingInfo
    cached: bool = False


# ── Extract Models ───────────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    """Request model for structured extraction."""
    url: HttpUrl
    schema_definition: dict
    instructions: Optional[str] = None

    @field_validator('url')
    @classmethod
    def validate_url_security(cls, v: HttpUrl) -> HttpUrl:
        return _validate_url(v)


class ExtractResponse(BaseModel):
    """Response model for structured extraction."""
    status: str
    data: Any
    confidence: float
    source_url: str
    char_count: int
    model_used: Optional[str] = None
    timing: TimingInfo
    cached: bool = False


# ── Batch Models ─────────────────────────────────────────────────────────────
class BatchRequest(BaseModel):
    """Request model for batch URL processing."""
    urls: list[HttpUrl]
    focus: Optional[str] = None

    @field_validator('urls')
    @classmethod
    def validate_urls(cls, v: list) -> list:
        if len(v) > 10:
            raise ValueError("Maximum 10 URLs per batch request")
        if len(v) == 0:
            raise ValueError("At least 1 URL is required")
        for url in v:
            _validate_url(url)
        return v


class BatchItem(BaseModel):
    """Single result within a batch response."""
    url: str
    status: str
    markdown: Optional[str] = None
    char_count: int = 0
    error: Optional[str] = None
    cached: bool = False


class BatchResponse(BaseModel):
    """Response model for batch processing."""
    status: str
    results: list[BatchItem]
    total_urls: int
    successful: int
    failed: int
    timing: TimingInfo


# ── Search Models ────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    """Request model for deep doc search."""
    url: HttpUrl
    query: str
    max_results: int = 3

    @field_validator('url')
    @classmethod
    def validate_url_security(cls, v: HttpUrl) -> HttpUrl:
        return _validate_url(v)

    @field_validator('max_results')
    @classmethod
    def validate_max_results(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError("max_results must be between 1 and 10")
        return v


class SearchResult(BaseModel):
    """A single search result."""
    section: str
    content: str
    relevance: float


class SearchResponse(BaseModel):
    """Response model for deep search."""
    status: str
    query: str
    results: list[SearchResult]
    source_url: str
    timing: TimingInfo
    cached: bool = False


# ── API Key Store ────────────────────────────────────────────────────────────
API_KEYS: dict[str, dict] = {}
API_KEYS_FILE = BASE_DIR / "api_keys.json"
_api_keys_lock = threading.Lock()


def load_api_keys():
    """Load API keys from file."""
    global API_KEYS
    if API_KEYS_FILE.exists():
        with _api_keys_lock:
            with open(API_KEYS_FILE) as f:
                API_KEYS = json.load(f)


def save_api_keys():
    """Persist API keys to file (thread-safe)."""
    with _api_keys_lock:
        with open(API_KEYS_FILE, "w") as f:
            json.dump(API_KEYS, f, indent=2)


def get_api_tier(api_key: Optional[str]) -> str:
    """Determine the tier for a request based on API key."""
    if not api_key:
        return "free"
    key_data = API_KEYS.get(api_key)
    if not key_data:
        return "free"
    return str(key_data.get("tier", "dev"))


# Tier-based rate limits (generous for launch — all features free)
TIER_RATE_LIMITS = {
    "free": "20/minute",
    "dev": "60/minute",
    "pro": "300/minute",
}


# ── FastAPI App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="Doc-Squeeze",
    version="1.0.0",
    description=(
        "Knowledge extraction API for AI agents. "
        "Fetch docs as markdown, extract structured JSON, batch-process URLs, "
        "and search within documents. Supports MCP, OpenAI plugin, and OpenClaw discovery."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "squeeze", "description": "Fetch documentation as clean markdown"},
        {"name": "extract", "description": "Structured data extraction from URLs"},
        {"name": "batch", "description": "Multi-URL parallel processing"},
        {"name": "search", "description": "Deep search within documents"},
        {"name": "cache", "description": "Cache management and stats"},
        {"name": "keys", "description": "API key management"},
        {"name": "discovery", "description": "Agent self-discovery endpoints"},
        {"name": "health", "description": "Service health monitoring"},
    ],
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore

# Gzip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-API-Key"],
)


# ── Global HTTP Client & Groq Client ─────────────────────────────────────────
http_client: Optional[httpx.AsyncClient] = None
groq_client: Optional[Groq] = None


# ── Exception Handlers ───────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standardized error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "request_failed",
            "detail": exc.detail,
            "code": f"ERR_{exc.status_code}"
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "detail": str(exc),
            "code": "ERR_400"
        }
    )


# ── Request timing + auth middleware ─────────────────────────────────────────
@app.middleware("http")
async def add_timing_and_logging(request: Request, call_next):
    """Add response time tracking, request ID, API key auth, and cache headers."""
    start = time.time()

    # Generate request ID for tracing
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.request_id = request_id

    # Extract API key and determine tier
    api_key = request.headers.get("X-API-Key")
    tier = get_api_tier(api_key)
    request.state.api_tier = tier

    # Track usage for authenticated keys
    if api_key and api_key in API_KEYS:
        API_KEYS[api_key]["requests"] = API_KEYS[api_key].get("requests", 0) + 1

    logger.info(
        "[%s] %s %s from %s (tier=%s)",
        request_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
        tier,
    )

    response = await call_next(request)

    elapsed = round(time.time() - start, 3)
    response.headers["X-Response-Time"] = f"{elapsed}s"
    response.headers["X-Request-ID"] = request_id
    response.headers["X-API-Tier"] = tier
    response.headers["X-RateLimit-Tier"] = TIER_RATE_LIMITS.get(tier, "5/minute")

    # Add cache header if set by endpoint
    cache_status = getattr(request.state, "cache_status", None)
    if cache_status:
        response.headers["X-Cache"] = cache_status

    logger.info(
        "[%s] %s %s → %s (%ss)%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        f" [CACHE {cache_status}]" if cache_status else "",
    )

    return response


# ── Lifespan (replaces deprecated on_event) ─────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup and shutdown lifecycle."""
    global http_client, groq_client

    logger.info("──────────────────────────────────────────")
    logger.info("  Doc-Squeeze v1.0.0 starting up")
    logger.info("  Environment: %s", settings.env)
    logger.info("  CORS Origins: %s", settings.cors_origins)
    logger.info("  GROQ_API_KEY: %s", "✅ configured" if settings.groq_api_key else "❌ not set (focus filtering disabled)")
    logger.info("  Model: %s", settings.groq_model)
    logger.info("  Rate Limit: %s", settings.rate_limit_squeeze)
    logger.info("  Cache TTL: %ds | Max: %d entries", settings.cache_ttl, settings.cache_max_entries)
    logger.info("──────────────────────────────────────────")

    # Initialize async HTTP client with connection pooling
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.jina_timeout),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
    )

    # Initialize Groq client if API key is available
    if settings.groq_api_key:
        groq_client = Groq(api_key=settings.groq_api_key)
        logger.info("Groq client initialized successfully")

    # Load API keys
    load_api_keys()
    logger.info("API keys loaded: %d keys", len(API_KEYS))

    yield  # ← app is running

    # Shutdown
    if http_client:
        await http_client.aclose()
        logger.info("HTTP client closed")
    # Persist API key usage stats
    save_api_keys()
    logger.info("API keys saved")


app.router.lifespan_context = lifespan


# ── Helpers ──────────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True
)
async def fetch_via_jina(target_url: str) -> str:
    """
    Fetch a URL's content as Markdown through the Jina Reader API.
    Includes retry logic with exponential backoff.
    """
    if not http_client:
        raise RuntimeError("HTTP client not initialized")

    jina_url = f"{settings.jina_prefix}{target_url}"
    logger.info("Fetching via Jina: %s", jina_url)

    try:
        resp = await http_client.get(
            jina_url,
            headers={"Accept": "text/markdown"},
            timeout=settings.jina_timeout
        )
        resp.raise_for_status()
        return resp.text

    except httpx.TimeoutException as exc:
        logger.error("Jina Reader timeout for URL: %s", target_url)
        raise HTTPException(
            status_code=504,
            detail="Jina Reader timed out while fetching the target URL. The page may be too large or slow to load.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.error("Jina Reader HTTP error %s for URL: %s", exc.response.status_code, target_url)
        raise HTTPException(
            status_code=502,
            detail=f"Jina Reader returned error {exc.response.status_code}. The URL may be inaccessible.",
        ) from exc
    except httpx.RequestError as exc:
        logger.error("Jina Reader request failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Jina Reader request failed: {type(exc).__name__}",
        ) from exc


async def filter_with_groq(markdown: str, focus: str) -> str:
    """
    Use Groq (Llama-3.3 70B) to extract only the parts relevant to `focus`.
    Returns original markdown if Groq is unavailable or fails.
    Runs in a thread to avoid blocking the event loop.
    """
    if not groq_client:
        logger.warning("GROQ_API_KEY not set — skipping intelligence filter")
        return markdown

    # Truncate to stay within context window limits
    truncated = markdown[:settings.groq_max_input_chars]

    logger.info("Filtering with Groq — focus: '%s' | input chars: %d", focus, len(truncated))

    def _sync_call():
        return groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a documentation filter. "
                        "Return ONLY the parts of the following markdown that are relevant to the user's focus topic. "
                        "Preserve original markdown formatting. Do not add commentary or explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"**Focus topic:** {focus}\n\n---\n\n{truncated}",
                },
            ],
            temperature=0.2,
            max_tokens=settings.groq_max_output_tokens,
        )

    try:
        chat = await asyncio.to_thread(_sync_call)

        filtered = chat.choices[0].message.content or markdown
        logger.info("Groq filtering successful — output chars: %d", len(filtered))
        return filtered

    except Exception as exc:
        logger.error("Groq filtering failed: %s — returning original markdown", exc)
        # Graceful degradation — return raw markdown on failure
        return markdown


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    """Health check for Render / uptime monitors."""
    return {
        "status": "healthy",
        "service": "doc-squeeze",
        "version": "1.0.0",
        "groq_configured": settings.groq_api_key is not None,
        "environment": settings.env,
        "cache": content_cache.stats(),
        "api_keys_loaded": len(API_KEYS),
    }


@app.get("/api/skill", tags=["discovery"])
async def skill_manifest():
    """
    Agent self-discovery — returns the full ClawHub skill manifest
    so agents can introspect available tools at runtime.
    """
    manifest_path = BASE_DIR / "openclaw.json"
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="openclaw.json manifest not found.") from exc


# ── Agent Discovery Endpoints ────────────────────────────────────────────────
@app.get("/.well-known/ai-plugin.json", tags=["discovery"])
async def ai_plugin_manifest(request: Request):
    """
    OpenAI-compatible plugin manifest.
    Allows ChatGPT, Langchain, and similar agents to auto-discover this tool.
    """
    base = str(request.base_url).rstrip("/")
    return {
        "schema_version": "v1",
        "name_for_human": "Doc-Squeeze",
        "name_for_model": "doc_squeeze",
        "description_for_human": "Knowledge extraction API — fetch docs, extract structured data, search, and batch-process URLs.",
        "description_for_model": (
            "Use this tool when you need to read documentation or extract data from web pages. "
            "Available operations: "
            "1) POST /api/squeeze — fetch URL as markdown, optional focus filter. "
            "2) POST /api/extract — extract structured JSON using a schema. "
            "3) POST /api/search — search within a page for specific answers. "
            "4) POST /api/batch — fetch multiple URLs in parallel. "
            "All responses include timing metadata and cache status."
        ),
        "auth": {"type": "none"},
        "api": {
            "type": "openapi",
            "url": f"{base}/openapi.json",
            "is_user_authenticated": False,
        },
        "logo_url": f"{base}/static/logo.png",
        "contact_email": "support@openclaw.dev",
        "legal_info_url": f"{base}/static/index.html",
    }


@app.get("/.well-known/mcp.json", tags=["discovery"])
async def mcp_discovery(request: Request):
    """
    MCP (Model Context Protocol) server discovery.
    Allows Cursor, Claude Desktop, Windsurf, and MCP-compatible clients
    to discover and connect to this tool.
    """
    base = str(request.base_url).rstrip("/")
    return {
        "name": "doc-squeeze",
        "version": "1.0.0",
        "description": "Knowledge extraction API for AI agents — fetch docs, extract structured data, batch-process, and search.",
        "transport": {
            "type": "http",
            "url": base,
        },
        "tools": [
            {
                "name": "squeeze_url",
                "description": "Fetch a documentation URL and return its content as clean markdown. Optionally filter to a specific topic.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "format": "uri", "description": "URL to read."},
                        "focus": {"type": "string", "description": "Optional topic filter."},
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "extract_structured",
                "description": "Extract structured JSON data from a URL using a schema. Returns typed data instead of raw markdown.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "format": "uri", "description": "URL to extract from."},
                        "schema_definition": {"type": "object", "description": "JSON schema defining the structure to extract."},
                        "instructions": {"type": "string", "description": "Optional extraction guidance."},
                    },
                    "required": ["url", "schema_definition"],
                },
            },
            {
                "name": "search_docs",
                "description": "Search within a documentation page for answers to a specific query. Returns relevant sections with relevance scores.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "format": "uri", "description": "URL to search within."},
                        "query": {"type": "string", "description": "What to search for."},
                        "max_results": {"type": "integer", "default": 3, "description": "Max results (1-10)."},
                    },
                    "required": ["url", "query"],
                },
            },
            {
                "name": "batch_squeeze",
                "description": "Fetch multiple URLs in parallel. Up to 10 URLs per request.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "urls": {"type": "array", "items": {"type": "string", "format": "uri"}, "description": "URLs to fetch (max 10)."},
                        "focus": {"type": "string", "description": "Optional topic filter applied to all."},
                    },
                    "required": ["urls"],
                },
            },
            {
                "name": "health_check",
                "description": "Check if Doc-Squeeze is running and healthy.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ],
        "instructions": (
            "Doc-Squeeze is a knowledge extraction API for AI agents. "
            "Use squeeze_url to fetch docs as markdown. Use extract_structured to get typed JSON data. "
            "Use search_docs to find specific answers within a page. Use batch_squeeze for multiple URLs. "
            "All endpoints support smart caching — repeated requests are sub-50ms. "
            "Example: extract_structured(url='https://docs.stripe.com/api', schema_definition={'endpoints': [{'method': 'str', 'path': 'str'}]})"
        ),
    }


@app.get("/", include_in_schema=False)
async def root():
    """Serve the interactive playground."""
    index_path = BASE_DIR / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"service": "doc-squeeze", "version": "1.0.0", "docs": "/docs"})


# ── Squeeze Endpoint ─────────────────────────────────────────────────────────
@app.post("/api/squeeze", response_model=SqueezeResponse, tags=["squeeze"])
@limiter.limit(settings.rate_limit_squeeze)
async def squeeze(request: Request, req: SqueezeRequest):
    """
    Fetch a documentation URL via Jina Reader and optionally filter
    the resulting Markdown to a specific focus topic using Groq.

    **Smart caching**: Repeated requests for the same URL return in <50ms.

    **Agent usage:**
    ```json
    POST /api/squeeze
    {"url": "https://docs.stripe.com/api", "focus": "error handling"}
    ```
    """
    target = str(req.url)
    logger.info("Squeeze request — url=%s | focus=%s", target, req.focus)
    request_start = time.time()

    # ── Check cache ──
    cache_key = SmartCache.make_key("squeeze", target, req.focus or "")
    cached = content_cache.get(cache_key)
    if cached:
        total_ms = int((time.time() - request_start) * 1000)
        cached["timing"] = TimingInfo(fetch_ms=0, filter_ms=0, total_ms=total_ms)
        cached["cached"] = True
        request.state.cache_status = "HIT"
        return SqueezeResponse(**cached)

    request.state.cache_status = "MISS"

    # 1. Fetch markdown via Jina Reader (with retry logic)
    fetch_start = time.time()
    raw_md = await fetch_via_jina(target)
    fetch_ms = int((time.time() - fetch_start) * 1000)

    if not raw_md or not raw_md.strip():
        raise HTTPException(
            status_code=422,
            detail="Jina returned empty content for the given URL. The page may not exist or may be blocking scrapers."
        )

    # 2. Optionally filter with Groq
    was_filtered = False
    final_md = raw_md
    filter_ms = None
    model_used = None

    if req.focus and groq_client:
        filter_start = time.time()
        final_md = await filter_with_groq(raw_md, req.focus)
        filter_ms = int((time.time() - filter_start) * 1000)
        was_filtered = True
        model_used = settings.groq_model

    total_ms = int((time.time() - request_start) * 1000)

    result = {
        "status": "success",
        "markdown": final_md,
        "source": "jina.ai",
        "char_count": len(final_md),
        "raw_char_count": len(raw_md),
        "was_filtered": was_filtered,
        "model_used": model_used,
    }
    content_cache.set(cache_key, result, ttl=settings.cache_ttl)

    return SqueezeResponse(
        status="success",
        markdown=final_md,
        source="jina.ai",
        char_count=len(final_md),
        raw_char_count=len(raw_md),
        was_filtered=was_filtered,
        model_used=model_used,
        timing=TimingInfo(fetch_ms=fetch_ms, filter_ms=filter_ms, total_ms=total_ms),
        cached=False,
    )


# ── Structured Extraction ────────────────────────────────────────────────────
@app.post("/api/extract", response_model=ExtractResponse, tags=["extract"])
@limiter.limit(settings.rate_limit_squeeze)
async def extract_structured(request: Request, req: ExtractRequest):
    """
    Extract structured JSON data from a URL using a schema definition.

    Instead of raw markdown, get typed data. Agent sends a URL + a JSON schema
    describing what they want, and gets back structured, parseable data.

    **Agent usage:**
    ```json
    POST /api/extract
    {
      "url": "https://docs.stripe.com/api",
      "schema_definition": {
        "endpoints": [{"method": "str", "path": "str", "description": "str"}],
        "auth_methods": ["str"]
      }
    }
    ```
    """
    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="Structured extraction requires Groq API key. Set GROQ_API_KEY in environment."
        )

    target = str(req.url)
    logger.info("Extract request — url=%s", target)
    request_start = time.time()

    # Check cache
    schema_str = json.dumps(req.schema_definition, sort_keys=True)
    cache_key = SmartCache.make_key("extract", target, schema_str)
    cached = content_cache.get(cache_key)
    if cached:
        total_ms = int((time.time() - request_start) * 1000)
        cached["timing"] = TimingInfo(fetch_ms=0, extract_ms=0, total_ms=total_ms)
        cached["cached"] = True
        request.state.cache_status = "HIT"
        return ExtractResponse(**cached)

    request.state.cache_status = "MISS"

    # 1. Fetch markdown
    fetch_start = time.time()
    raw_md = await fetch_via_jina(target)
    fetch_ms = int((time.time() - fetch_start) * 1000)

    if not raw_md or not raw_md.strip():
        raise HTTPException(status_code=422, detail="Empty content from URL.")

    # 2. Extract structured data via Groq
    extract_start = time.time()
    truncated = raw_md[:settings.groq_max_input_chars]

    extra_instructions = ""
    if req.instructions:
        extra_instructions = f"\nAdditional instructions: {req.instructions}"

    def _sync_extract():
        return groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data extraction engine. Given a document and a JSON schema, "
                        "extract the data that matches the schema from the document. "
                        "Return ONLY valid JSON that conforms to the schema. "
                        "No commentary, no markdown fences, just the JSON object."
                        f"{extra_instructions}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"**Schema to extract:**\n```json\n{schema_str}\n```\n\n"
                        f"**Document:**\n{truncated}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=settings.groq_max_output_tokens,
        )

    try:
        chat = await asyncio.to_thread(_sync_extract)

        raw_output = chat.choices[0].message.content or "{}"
        # Strip markdown fences if the model added them
        raw_output = raw_output.strip()
        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n", 1)[-1]
        if raw_output.endswith("```"):
            raw_output = raw_output.rsplit("```", 1)[0]
        raw_output = raw_output.strip()

        extracted_data = json.loads(raw_output)

        # Dynamic confidence: check how many schema keys got populated
        schema_keys = set(req.schema_definition.keys())
        result_keys = set(extracted_data.keys()) if isinstance(extracted_data, dict) else set()
        match_ratio = len(schema_keys & result_keys) / max(len(schema_keys), 1)
        confidence = round(0.5 + (match_ratio * 0.45), 2)  # 0.50 - 0.95 range
    except json.JSONDecodeError:
        extracted_data = {"raw_text": raw_output, "parse_error": "LLM output was not valid JSON"}
        confidence = 0.3
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Extraction failed: {exc}") from exc

    extract_ms = int((time.time() - extract_start) * 1000)
    total_ms = int((time.time() - request_start) * 1000)

    result = {
        "status": "success",
        "data": extracted_data,
        "confidence": confidence,
        "source_url": target,
        "char_count": len(raw_md),
        "model_used": settings.groq_model,
    }
    content_cache.set(cache_key, result, ttl=settings.cache_ttl)

    result["timing"] = TimingInfo(fetch_ms=fetch_ms, extract_ms=extract_ms, total_ms=total_ms)
    result["cached"] = False
    return ExtractResponse(**result)


# ── Batch Processing ─────────────────────────────────────────────────────────
@app.post("/api/batch", response_model=BatchResponse, tags=["batch"])
@limiter.limit(settings.rate_limit_squeeze)
async def batch_squeeze(request: Request, req: BatchRequest):
    """
    Fetch multiple URLs in parallel. Up to 10 URLs per request.

    **Agent usage:**
    ```json
    POST /api/batch
    {
      "urls": ["https://docs.stripe.com/api/authentication", "https://docs.stripe.com/api/errors"],
      "focus": "error handling"
    }
    ```
    """
    logger.info("Batch request — %d URLs", len(req.urls))
    request_start = time.time()

    async def process_single(url: HttpUrl) -> BatchItem:
        target = str(url)
        try:
            # Check cache
            cache_key = SmartCache.make_key("squeeze", target, req.focus or "")
            cached = content_cache.get(cache_key)
            if cached:
                return BatchItem(
                    url=target,
                    status="success",
                    markdown=cached["markdown"],
                    char_count=cached["char_count"],
                    cached=True,
                )

            raw_md = await fetch_via_jina(target)
            if not raw_md or not raw_md.strip():
                return BatchItem(url=target, status="error", error="Empty content")

            final_md = raw_md
            if req.focus and groq_client:
                final_md = await filter_with_groq(raw_md, req.focus)

            # Cache individual result
            result_data = {
                "status": "success",
                "markdown": final_md,
                "source": "jina.ai",
                "char_count": len(final_md),
                "raw_char_count": len(raw_md),
                "was_filtered": bool(req.focus and groq_client),
                "model_used": settings.groq_model if req.focus and groq_client else None,
            }
            content_cache.set(cache_key, result_data, ttl=settings.cache_ttl)

            return BatchItem(
                url=target,
                status="success",
                markdown=final_md,
                char_count=len(final_md),
                cached=False,
            )
        except HTTPException as e:
            return BatchItem(url=target, status="error", error=e.detail)
        except Exception as e:
            return BatchItem(url=target, status="error", error=str(e))

    # Run all URLs in parallel
    fetch_start = time.time()
    results = await asyncio.gather(*[process_single(url) for url in req.urls])
    fetch_ms = int((time.time() - fetch_start) * 1000)

    total_ms = int((time.time() - request_start) * 1000)
    successful = sum(1 for r in results if r.status == "success")

    request.state.cache_status = "BATCH"

    return BatchResponse(
        status="success",
        results=list(results),
        total_urls=len(req.urls),
        successful=successful,
        failed=len(req.urls) - successful,
        timing=TimingInfo(fetch_ms=fetch_ms, total_ms=total_ms),
    )


# ── Deep Search ──────────────────────────────────────────────────────────────
@app.post("/api/search", response_model=SearchResponse, tags=["search"])
@limiter.limit(settings.rate_limit_squeeze)
async def search_docs(request: Request, req: SearchRequest):
    """
    Search within a documentation page for specific answers.

    Fetches the page, then uses an LLM to find the most relevant sections
    that answer the query. Returns ranked results with relevance scores.

    **Agent usage:**
    ```json
    POST /api/search
    {
      "url": "https://docs.python.org/3/library/asyncio.html",
      "query": "How do I run multiple coroutines concurrently?",
      "max_results": 3
    }
    ```
    """
    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="Search requires Groq API key. Set GROQ_API_KEY in environment."
        )

    target = str(req.url)
    logger.info("Search request — url=%s | query=%s", target, req.query)
    request_start = time.time()

    # Check cache
    cache_key = SmartCache.make_key("search", target, req.query, str(req.max_results))
    cached = content_cache.get(cache_key)
    if cached:
        total_ms = int((time.time() - request_start) * 1000)
        cached["timing"] = TimingInfo(fetch_ms=0, total_ms=total_ms)
        cached["cached"] = True
        request.state.cache_status = "HIT"
        return SearchResponse(**cached)

    request.state.cache_status = "MISS"

    # 1. Fetch markdown
    fetch_start = time.time()
    raw_md = await fetch_via_jina(target)
    fetch_ms = int((time.time() - fetch_start) * 1000)

    if not raw_md or not raw_md.strip():
        raise HTTPException(status_code=422, detail="Empty content from URL.")

    # 2. Search via Groq
    truncated = raw_md[:settings.groq_max_input_chars]

    def _sync_search():
        return groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a documentation search engine. Find the {req.max_results} most relevant "
                        "sections from the document that answer the user's query. "
                        "Return ONLY a JSON array of objects, each with: "
                        '"section" (section title or heading), '
                        '"content" (the relevant text, keep it concise but complete), '
                        '"relevance" (0.0 to 1.0 score). '
                        "Sort by relevance descending. No markdown fences, just JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": f"**Query:** {req.query}\n\n**Document:**\n{truncated}",
                },
            ],
            temperature=0.1,
            max_tokens=settings.groq_max_output_tokens,
        )

    try:
        chat = await asyncio.to_thread(_sync_search)

        raw_output = chat.choices[0].message.content or "[]"
        raw_output = raw_output.strip()
        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n", 1)[-1]
        if raw_output.endswith("```"):
            raw_output = raw_output.rsplit("```", 1)[0]
        raw_output = raw_output.strip()

        search_results = json.loads(raw_output)
        if not isinstance(search_results, list):
            search_results = [search_results]

        results = [
            SearchResult(
                section=r.get("section", "Unknown"),
                content=r.get("content", ""),
                relevance=min(1.0, max(0.0, float(r.get("relevance", 0.5)))),
            )
            for r in search_results[:req.max_results]
        ]
    except json.JSONDecodeError:
        results = [SearchResult(section="Full Document", content=raw_output[:500], relevance=0.5)]
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc

    total_ms = int((time.time() - request_start) * 1000)

    result = {
        "status": "success",
        "query": req.query,
        "results": [r.model_dump() for r in results],
        "source_url": target,
    }
    content_cache.set(cache_key, result, ttl=settings.cache_ttl)


    return SearchResponse(
        status="success",
        query=req.query,
        results=results,
        source_url=target,
        timing=TimingInfo(fetch_ms=fetch_ms, total_ms=total_ms),
        cached=False,
    )


# ── Cache Management ─────────────────────────────────────────────────────────
@app.get("/api/cache/stats", tags=["cache"])
async def cache_stats():
    """Return cache statistics — entries, hit rate, uptime."""
    return content_cache.stats()


@app.delete("/api/cache", tags=["cache"])
async def cache_clear(request: Request):
    """Clear all cached entries. Requires an API key."""
    api_key = request.headers.get("X-API-Key")
    tier = get_api_tier(api_key)
    if tier == "free":
        raise HTTPException(
            status_code=403,
            detail="Cache management requires an API key. Generate one at POST /api/keys/create"
        )
    count = content_cache.clear()
    return {"cleared": count, "message": f"Cleared {count} cached entries"}


# ── API Key Management ───────────────────────────────────────────────────────
@app.post("/api/keys/create", tags=["keys"])
async def create_api_key(request: Request):
    """
    Generate a free API key. Returns the key — store it safely, it cannot be retrieved again.

    All features are free. Keys give you higher rate limits (60/min vs 20/min).
    """
    key = f"ds_{uuid.uuid4().hex[:24]}"
    API_KEYS[key] = {
        "tier": "dev",
        "created_at": time.time(),
        "requests": 0,
    }
    save_api_keys()

    return {
        "api_key": key,
        "tier": "dev",
        "rate_limit": TIER_RATE_LIMITS["dev"],
        "features": ["squeeze", "extract", "batch", "search"],
        "note": "All features are free. Key gives you 3x higher rate limits.",
        "usage": "Pass as X-API-Key header in requests.",
        "message": "Store this key safely — it cannot be retrieved again.",
    }


# ── Mount Static Files ───────────────────────────────────────────────────────
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.is_development,
    )
