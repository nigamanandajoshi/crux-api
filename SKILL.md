---
name: doc-squeeze-free
description: "Reads any documentation URL and returns clean, filtered markdown for AI agents. Powered by Jina Reader + Groq LLM — 100% free tier."
---

# Doc-Squeeze — ClawHub Skill

## Overview

Doc-Squeeze lets AI agents **read external documentation** without a browser. It fetches any URL as clean markdown and optionally focuses the content to a specific topic using an LLM.

> **Cost: $0** — Both Jina Reader and Groq have generous free tiers.

## Tools

### `squeeze_url`

| Property    | Value                     |
|-------------|---------------------------|
| **Endpoint**| `POST /api/squeeze`       |
| **Auth**    | None (public)             |
| **Latency** | ~2-5s (fetch) + ~1-3s (filter) |

**Input:**

```json
{
  "url": "https://docs.stripe.com/api/authentication",
  "focus": "Python code for setting the API key"
}
```

| Field   | Type   | Required | Description                                    |
|---------|--------|----------|------------------------------------------------|
| `url`   | string | ✅       | Full URL of the documentation page.            |
| `focus` | string | ❌       | Topic filter — triggers LLM extraction.        |

**Output:**

```json
{
  "status": "success",
  "markdown": "```python\nimport stripe\nstripe.api_key = 'sk_test_...'\n```",
  "source": "jina.ai",
  "char_count": 100,
  "was_filtered": true
}
```

| Field          | Type    | Description                              |
|----------------|---------|------------------------------------------|
| `status`       | string  | Always `"success"` on 200.               |
| `markdown`     | string  | Clean markdown content.                  |
| `source`       | string  | Upstream provider (`"jina.ai"`).         |
| `char_count`   | integer | Character count of returned markdown.    |
| `was_filtered` | boolean | Whether LLM filtering was applied.       |

**Errors:**

| Code | Meaning                         |
|------|---------------------------------|
| 422  | Invalid URL or empty response.  |
| 502  | Jina Reader request failed.     |
| 504  | Jina Reader timed out.          |

### Self-Discovery

Agents can introspect the full tool schema at runtime:

```
GET /api/skill → returns openclaw.json manifest
```

## Permissions

| Permission     | Host             | Required | Reason                          |
|----------------|------------------|----------|---------------------------------|
| Network Access | `r.jina.ai`      | ✅       | Fetches docs as markdown.       |
| Network Access | `api.groq.com`   | ❌       | LLM filtering (only with focus) |

## Environment

| Variable       | Required | How to get it                                        |
|----------------|----------|------------------------------------------------------|
| `GROQ_API_KEY` | No*      | Free at [console.groq.com/keys](https://console.groq.com/keys) |

> \*Without the key, focus filtering is skipped — raw markdown is returned instead.

## Agent Usage Example

```python
import requests

# Basic fetch — get full docs as markdown
resp = requests.post("https://doc-squeeze.onrender.com/api/squeeze", json={
    "url": "https://docs.python.org/3/library/json.html"
})
docs = resp.json()["markdown"]

# Focused fetch — only get what you need
resp = requests.post("https://doc-squeeze.onrender.com/api/squeeze", json={
    "url": "https://docs.python.org/3/library/json.html",
    "focus": "json.dumps parameters"
})
filtered = resp.json()["markdown"]
```
