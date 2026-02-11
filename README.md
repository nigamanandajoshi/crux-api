<div align="center">

# 🗜️ Doc-Squeeze

**Knowledge extraction API for AI agents.**

*Fetch docs, extract structured JSON, search within pages, batch-process URLs.*

*100% free — powered by [Jina Reader](https://jina.ai/reader/) + [Groq](https://groq.com/)*

[![ClawHub Skill](https://img.shields.io/badge/ClawHub-doc--squeeze-blueviolet?style=flat-square)](https://clawhub.dev)
[![MIT License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)

</div>

---

## What It Does

Doc-Squeeze is an API that lets AI agents **read any URL**, **extract structured data**, **search within documents**, and **batch-process multiple pages** — all through simple JSON endpoints with smart caching.

```
Agent → "Read Stripe docs, extract the API endpoints as JSON"
Doc-Squeeze → { "endpoints": [{"method": "POST", "path": "/v1/charges"}...] } ✅
```

## Quick Start

```bash
git clone https://github.com/your-username/ClawSearch.git
cd ClawSearch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add your free Groq key for LLM features
cp .env.example .env   # then edit .env with your key

python main.py
# → http://localhost:8000
```

## API Endpoints

### Core

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/squeeze` | POST | Fetch URL as clean markdown (optional focus filter) |
| `/api/extract` | POST | Extract structured JSON using a schema definition |
| `/api/search` | POST | Deep search within a document for specific answers |
| `/api/batch` | POST | Fetch multiple URLs in parallel (max 10) |

### Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cache/stats` | GET | Cache hit rate, entries, uptime |
| `/api/cache` | DELETE | Clear cache (requires API key) |
| `/api/keys/create` | POST | Generate an API key |

### Discovery

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/skill` | GET | OpenClaw skill manifest |
| `/.well-known/ai-plugin.json` | GET | OpenAI plugin manifest |
| `/.well-known/mcp.json` | GET | MCP server discovery |
| `/docs` | GET | Swagger UI |
| `/health` | GET | Health check |

## Usage Examples

### Squeeze (Raw or Filtered)
```bash
curl -X POST http://localhost:8000/api/squeeze \
  -H "Content-Type: application/json" \
  -d '{"url":"https://docs.stripe.com/api/authentication", "focus":"Python API key setup"}'
```

### Structured Extraction
```bash
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.stripe.com/api",
    "schema_definition": {"endpoints": [{"method": "str", "path": "str"}]},
    "instructions": "Focus on payment endpoints"
  }'
```

### Deep Search
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.python.org/3/library/asyncio.html", "query": "cancel a task"}'
```

### Batch Processing
```bash
curl -X POST http://localhost:8000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com", "https://httpbin.org/html"], "focus": "main content"}'
```

## Authentication

| Tier | Rate Limit | Access |
|------|-----------|--------|
| **Free** | 5/minute | No key needed |
| **Dev** | 60/minute | `POST /api/keys/create` → use `X-API-Key` header |
| **Pro** | 300/minute | Contact us |

## Architecture

```
Agent ─── POST /api/* ──▶  Doc-Squeeze (FastAPI)
                               │
                    ┌──────────┴──────────┐
                    │   Smart Cache       │
                    │   (TTL: 15min)      │
                    └──────────┬──────────┘
                          HIT? │
                     ┌────yes──┴──no────┐
                     ▼                  ▼
                 < 50ms            GET r.jina.ai/{url}
                 cached                 │
                 response               ▼
                                   Raw Markdown
                                        │
                          ┌─────────────┴─────────────┐
                          │ /squeeze   /extract   /search │
                          │ (filter)   (schema)   (query) │
                          └─────────────┬─────────────┘
                                        │
                                   Groq LLM (llama-3.3-70b)
                                        │
                                        ▼
                                  JSON Response
```

## Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your repo — `render.yaml` handles everything
4. Add `GROQ_API_KEY` in **Environment** settings
5. Set `ENV=production` and `ALLOWED_ORIGINS=https://your-domain.com`
6. Done — Render auto-deploys on push

## Environment Variables

| Variable | Required | Default | Source |
|----------|----------|---------|--------|
| `GROQ_API_KEY` | No* | – | [console.groq.com/keys](https://console.groq.com/keys) (free) |
| `PORT` | No | 8000 | Auto-set by Render |
| `ENV` | No | development | `production` for deploy |
| `CACHE_TTL` | No | 900 | Seconds |
| `CACHE_MAX_ENTRIES` | No | 500 | Max cached items |

> *Without the key, only raw squeeze and batch work. Extract and search require Groq.

## Project Structure

```
├── main.py            # FastAPI app (all endpoints, middleware, auth)
├── cache.py           # Smart caching layer (TTL, LRU, stats)
├── mcp_server.py      # MCP server for Claude Desktop / Cursor
├── openclaw.json      # ClawHub skill manifest
├── SKILL.md           # Skill docs for agents
├── static/index.html  # Interactive playground
├── requirements.txt   # Python deps
├── render.yaml        # Render deploy blueprint
├── Dockerfile         # Container build
├── .env.example       # Env template
└── tests/             # Unit tests
```

## License

[MIT](LICENSE)
