---
name: doc-squeeze
description: "Knowledge extraction API for AI agents. Fetch docs as markdown, extract structured JSON, batch-process URLs, and search within documents. Powered by Jina Reader + Groq LLM — 100% free tier."
---

# Doc-Squeeze — ClawHub Skill

## Overview

Doc-Squeeze lets AI agents **read external documentation** and **extract structured knowledge** without a browser. It fetches any URL as clean markdown, extracts structured JSON from schema definitions, searches within documents, and processes multiple URLs in parallel.

> **Cost: $0** — Both Jina Reader and Groq have generous free tiers.

## Tools

### `squeeze_url`

| Property    | Value                        |
|-------------|------------------------------|
| **Endpoint**| `POST /api/squeeze`          |
| **Auth**    | Optional (API key for higher limits) |
| **Latency** | ~2-5s (fetch) + ~1-3s (filter) |

**Input:**
```json
{
  "url": "https://docs.stripe.com/api/authentication",
  "focus": "Python code for setting the API key"
}
```

| Field   | Type   | Required | Description                        |
|---------|--------|----------|------------------------------------|
| `url`   | string | ✅       | URL of the documentation page.     |
| `focus` | string | ❌       | Topic filter — triggers LLM.      |

---

### `extract_structured`

| Property    | Value                        |
|-------------|------------------------------|
| **Endpoint**| `POST /api/extract`          |
| **Auth**    | Optional                     |
| **Latency** | ~3-8s                        |

**Input:**
```json
{
  "url": "https://docs.stripe.com/api",
  "schema_definition": {
    "endpoints": [{"method": "str", "path": "str", "description": "str"}],
    "auth_methods": ["str"]
  },
  "instructions": "Focus on the payments API only"
}
```

| Field              | Type   | Required | Description                              |
|--------------------|--------|----------|------------------------------------------|
| `url`              | string | ✅       | URL to extract from.                     |
| `schema_definition`| object | ✅       | JSON schema defining what to extract.    |
| `instructions`     | string | ❌       | Additional extraction guidance.          |

---

### `search_docs`

| Property    | Value                        |
|-------------|------------------------------|
| **Endpoint**| `POST /api/search`           |
| **Auth**    | Optional                     |
| **Latency** | ~3-8s                        |

**Input:**
```json
{
  "url": "https://docs.python.org/3/library/asyncio.html",
  "query": "How do I run multiple coroutines concurrently?",
  "max_results": 3
}
```

| Field        | Type    | Required | Description                           |
|--------------|---------|----------|---------------------------------------|
| `url`        | string  | ✅       | URL to search within.                 |
| `query`      | string  | ✅       | What to search for.                   |
| `max_results`| integer | ❌       | Number of results (1-10, default 3).  |

---

### `batch_squeeze`

| Property    | Value                        |
|-------------|------------------------------|
| **Endpoint**| `POST /api/batch`            |
| **Auth**    | Optional                     |
| **Latency** | ~3-15s (parallel)            |

**Input:**
```json
{
  "urls": [
    "https://docs.stripe.com/api/authentication",
    "https://docs.stripe.com/api/errors"
  ],
  "focus": "error handling"
}
```

| Field   | Type   | Required | Description                            |
|---------|--------|----------|----------------------------------------|
| `urls`  | array  | ✅       | URLs to fetch (max 10).               |
| `focus` | string | ❌       | Optional topic filter for all URLs.    |

---

### Self-Discovery

Agents can introspect the full tool schema at runtime:

```
GET /api/skill          → openclaw.json manifest
GET /.well-known/mcp.json → MCP server discovery
GET /.well-known/ai-plugin.json → OpenAI plugin manifest
```

## Authentication

| Tier   | Rate Limit | How to Get |
|--------|-----------|------------|
| Free   | 5/minute  | No key needed |
| Dev    | 60/minute | `POST /api/keys/create` |
| Pro    | 300/minute| Contact us |

API keys are passed via `X-API-Key` header.

## Permissions

| Permission     | Host             | Required | Reason                          |
|----------------|------------------|----------|---------------------------------|
| Network Access | `r.jina.ai`      | ✅       | Fetches docs as markdown.       |
| Network Access | `api.groq.com`   | ❌       | LLM filtering (only with focus) |

## Environment

| Variable       | Required | How to get it                                        |
|----------------|----------|------------------------------------------------------|
| `GROQ_API_KEY` | No*      | Free at [console.groq.com/keys](https://console.groq.com/keys) |

> \*Without the key, only raw squeeze and batch work. Extract and search require Groq.

## Agent Usage Example

```python
import requests

BASE = "https://doc-squeeze.onrender.com"

# 1. Raw fetch
resp = requests.post(f"{BASE}/api/squeeze", json={
    "url": "https://docs.python.org/3/library/json.html"
})
docs = resp.json()["markdown"]

# 2. Structured extraction
resp = requests.post(f"{BASE}/api/extract", json={
    "url": "https://docs.stripe.com/api",
    "schema_definition": {
        "endpoints": [{"method": "str", "path": "str"}],
        "auth_type": "str"
    }
})
data = resp.json()["data"]

# 3. Deep search
resp = requests.post(f"{BASE}/api/search", json={
    "url": "https://docs.python.org/3/library/asyncio.html",
    "query": "How to cancel a task?"
})
answers = resp.json()["results"]

# 4. Batch fetch
resp = requests.post(f"{BASE}/api/batch", json={
    "urls": ["https://example.com", "https://httpbin.org/html"],
    "focus": "main content"
})
results = resp.json()["results"]
```
