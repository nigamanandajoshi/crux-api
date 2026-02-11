<div align="center">

# 🗜️ Doc-Squeeze

**Let AI agents read any documentation in seconds.**

*100% free — powered by [Jina Reader](https://jina.ai/reader/) + [Groq](https://groq.com/)*

[![ClawHub Skill](https://img.shields.io/badge/ClawHub-doc--squeeze--free-blueviolet?style=flat-square)](https://clawhub.dev)
[![MIT License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)

</div>

---

## What It Does

Doc-Squeeze is a single-endpoint API that **fetches any URL as clean markdown** and optionally uses an LLM to **extract only the sections you need**.

```
Agent → "Read the Stripe docs, just the Python auth code"
Doc-Squeeze → 100 chars of exactly what was asked for ✅
```

## Quick Start

```bash
git clone https://github.com/your-username/ClawSearch.git
cd ClawSearch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: add your free Groq key for focus filtering
cp .env.example .env   # then edit .env with your key

python main.py
# → http://localhost:8000
```

## API

### `POST /api/squeeze`

| Param   | Type   | Required | Description                        |
|---------|--------|----------|------------------------------------|
| `url`   | string | ✅       | Documentation URL to read          |
| `focus` | string | –        | Topic filter (triggers LLM)        |

```bash
curl -X POST http://localhost:8000/api/squeeze \
  -H "Content-Type: application/json" \
  -d '{"url":"https://docs.stripe.com/api/authentication", "focus":"Python API key setup"}'
```

```json
{
  "status": "success",
  "markdown": "```python\nimport stripe\nstripe.api_key = 'sk_test_...'\n```",
  "source": "jina.ai",
  "char_count": 100,
  "was_filtered": true
}
```

### Other Endpoints

| Route          | Method | Purpose                          |
|----------------|--------|----------------------------------|
| `/`            | GET    | Landing page                     |
| `/health`      | GET    | Health check (Render/monitoring) |
| `/api/skill`   | GET    | Agent self-discovery manifest    |
| `/docs`        | GET    | Swagger UI (auto-generated)      |

## Architecture

```
Agent  ─── POST /api/squeeze ──▶  Doc-Squeeze (FastAPI)
                                       │
                               GET r.jina.ai/{url}
                                       │
                                       ▼
                                  Raw Markdown
                                       │
                           ┌───────────┴───────────┐
                           │  focus param provided? │
                           └───────────┬───────────┘
                              yes      │      no
                               ▼       │       ▼
                         Groq LLM      │   return raw
                       (llama-3.3-70b) │
                               ▼       │
                       Filtered MD  ───┘──▶  JSON response
```

## Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your repo — `render.yaml` handles everything
4. Add `GROQ_API_KEY` in **Environment** settings
5. Done — Render auto-deploys on push

## Environment Variables

| Variable       | Required | Source |
|----------------|----------|--------|
| `GROQ_API_KEY` | No*      | [console.groq.com/keys](https://console.groq.com/keys) (free) |
| `PORT`         | No       | Auto-set by Render |

> *Without the key, the API still works — it just skips focus-based filtering.

## Project Structure

```
├── main.py            # FastAPI app (all endpoints)
├── openclaw.json      # ClawHub skill manifest
├── SKILL.md           # Skill docs for agents
├── requirements.txt   # Python deps
├── render.yaml        # Render deploy blueprint
├── Dockerfile         # Container build
├── .env.example       # Env template
├── .dockerignore      # Docker exclusions
└── LICENSE            # MIT
```

## License

[MIT](LICENSE)
