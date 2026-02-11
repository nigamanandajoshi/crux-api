"""
Doc-Squeeze API — Free-tier documentation reader for AI agents.
Stack: FastAPI + Jina Reader (scraping) + Groq/Llama3 (intelligence).
"""

import os
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl
from groq import Groq

# ── Bootstrap ────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
logger = logging.getLogger("doc-squeeze")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JINA_PREFIX = "https://r.jina.ai/"
JINA_TIMEOUT = 30          # seconds
MAX_RAW_CHARS = 20_000     # threshold before Groq filtering kicks in
GROQ_MODEL = "llama-3.3-70b-versatile"

app = FastAPI(
    title="Doc-Squeeze",
    version="0.1.0",
    description="Lightweight API that lets AI agents read external documentation.",
)

# ── CORS (agents call from anywhere) ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ────────────────────────────────────────────────
@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round(time.time() - start, 3)
    response.headers["X-Response-Time"] = f"{elapsed}s"
    logger.info("%s %s → %s (%.3fs)", request.method, request.url.path, response.status_code, elapsed)
    return response


# ── Startup event ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_check():
    logger.info("──────────────────────────────────────────")
    logger.info("  Doc-Squeeze v0.1.0 starting up")
    logger.info("  GROQ_API_KEY: %s", "✅ configured" if GROQ_API_KEY else "❌ not set (focus filtering disabled)")
    logger.info("  Model: %s", GROQ_MODEL)
    logger.info("──────────────────────────────────────────")


# ── Models ───────────────────────────────────────────────────────────────────
class SqueezeRequest(BaseModel):
    url: HttpUrl
    focus: Optional[str] = None


class SqueezeResponse(BaseModel):
    status: str
    markdown: str
    source: str
    char_count: int
    was_filtered: bool


# ── Helpers ──────────────────────────────────────────────────────────────────
def fetch_via_jina(target_url: str) -> str:
    """Fetch a URL's content as Markdown through the Jina Reader API."""
    jina_url = f"{JINA_PREFIX}{target_url}"
    logger.info("Fetching via Jina: %s", jina_url)
    try:
        resp = requests.get(jina_url, timeout=JINA_TIMEOUT, headers={
            "Accept": "text/markdown",
        })
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Jina Reader timed out while fetching the target URL.",
        )
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Jina Reader request failed: {exc}",
        )
    return resp.text


def filter_with_groq(markdown: str, focus: str) -> str:
    """Use Groq (Llama-3.3 70B) to extract only the parts relevant to `focus`."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — skipping intelligence filter.")
        return markdown

    client = Groq(api_key=GROQ_API_KEY)

    # Truncate to ~60k chars to stay within context window limits
    truncated = markdown[:60_000]

    logger.info("Filtering with Groq — focus: '%s' | input chars: %d", focus, len(truncated))
    try:
        chat = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a documentation filter. "
                        "Return ONLY the parts of the following markdown that are relevant to the user's focus topic. "
                        "Preserve original markdown formatting. Do not add commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": f"**Focus topic:** {focus}\n\n---\n\n{truncated}",
                },
            ],
            temperature=0.2,
            max_tokens=8192,
        )
        return chat.choices[0].message.content or markdown
    except Exception as exc:
        logger.error("Groq filtering failed: %s", exc)
        # Graceful degradation — return raw markdown on failure
        return markdown


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check for Render / uptime monitors."""
    return {
        "status": "healthy",
        "service": "doc-squeeze",
        "version": "0.1.0",
        "groq_configured": GROQ_API_KEY is not None,
    }


@app.get("/api/skill")
async def skill_manifest():
    """
    Agent self-discovery — returns the full ClawHub skill manifest
    so agents can introspect available tools at runtime.
    """
    import json
    manifest_path = os.path.join(os.path.dirname(__file__), "openclaw.json")
    try:
        with open(manifest_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openclaw.json manifest not found.")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple welcome page."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Doc-Squeeze API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
                min-height: 100vh;
                display: flex; align-items: center; justify-content: center;
                background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                color: #e0e0e0;
            }
            .card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 3rem 2.5rem;
                max-width: 540px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            }
            .emoji { font-size: 3rem; margin-bottom: 1rem; }
            h1 {
                font-size: 2rem; font-weight: 700;
                background: linear-gradient(90deg, #a78bfa, #7dd3fc);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            .tagline { color: #a0a0b8; margin-bottom: 2rem; font-size: 0.95rem; }
            code {
                display: inline-block;
                background: rgba(255,255,255,0.08);
                padding: 0.15rem 0.6rem;
                border-radius: 6px;
                font-family: 'Fira Code', 'Cascadia Code', monospace;
                font-size: 0.85rem;
                color: #7dd3fc;
            }
            .endpoint {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 1.25rem;
                margin-top: 1.5rem;
                text-align: left;
                font-size: 0.88rem;
                line-height: 1.6;
            }
            .endpoint strong { color: #a78bfa; }
            .badge {
                display: inline-block;
                background: #7dd3fc22;
                color: #7dd3fc;
                font-size: 0.7rem;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 600;
                margin-bottom: 1rem;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="emoji">🗜️</div>
            <span class="badge">OpenClaw · Free Tier</span>
            <h1>Doc-Squeeze</h1>
            <p class="tagline">Lightweight doc reader for AI agents — powered by Jina&nbsp;+&nbsp;Groq.</p>
            <div class="endpoint">
                <strong>POST</strong> <code>/api/squeeze</code><br><br>
                <strong>Body:</strong><br>
                <code>{ "url": "https://...", "focus": "optional" }</code><br><br>
                <strong>Returns:</strong> Filtered markdown from any documentation URL.
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/api/squeeze", response_model=SqueezeResponse)
async def squeeze(req: SqueezeRequest):
    """
    Fetch a documentation URL via Jina Reader and optionally filter
    the resulting Markdown to a specific focus topic using Groq.
    """
    target = str(req.url)
    logger.info("Squeeze request — url=%s | focus=%s", target, req.focus)

    # 1. Fetch markdown via Jina Reader
    raw_md = fetch_via_jina(target)

    if not raw_md or not raw_md.strip():
        raise HTTPException(status_code=422, detail="Jina returned empty content for the given URL.")

    # 2. Optionally filter with Groq
    was_filtered = False
    final_md = raw_md

    if req.focus:
        if len(raw_md) > MAX_RAW_CHARS:
            # Content is long — Groq filter will add real value
            final_md = filter_with_groq(raw_md, req.focus)
            was_filtered = True
        else:
            # Content is short enough — still filter if focus is provided
            final_md = filter_with_groq(raw_md, req.focus)
            was_filtered = True

    return SqueezeResponse(
        status="success",
        markdown=final_md,
        source="jina.ai",
        char_count=len(final_md),
        was_filtered=was_filtered,
    )


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    is_dev = os.environ.get("ENV", "development") == "development"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=is_dev,
    )
