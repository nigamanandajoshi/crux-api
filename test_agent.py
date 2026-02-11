"""
🤖 Agent Simulation Test Suite for Doc-Squeeze v1.0.0
========================================================
Simulates what a real AI agent does when consuming the API:
  1. Discovers the skill (GET /api/skill)
  2. Checks health (GET /health)
  3. Fetches raw docs (POST /api/squeeze without focus)
  4. Fetches filtered docs (POST /api/squeeze with focus)
  5. Tests structured extraction (POST /api/extract)
  6. Tests deep search (POST /api/search)
  7. Tests batch processing (POST /api/batch)
  8. Tests cache stats and management
  9. Tests API key creation and auth headers
  10. Tests URL validation (security)
  11. Tests error response format

Run:  python test_agent.py
"""

import os
import sys
import time

import requests

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
PASS = "✅"
FAIL = "❌"
results = []


def test(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    print(f"  {status} {name}")
    if detail:
        print(f"     ↳ {detail}")
    print()


def main():
    print()
    print("=" * 60)
    print("🤖  DOC-SQUEEZE v1.0.0 — Agent Simulation Test Suite")
    print("=" * 60)
    print()

    # ── Test 1: Health Check ──────────────────────────────────
    print("─── Test 1: Health Check ───")
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        data = r.json()
        test(
            "Server is healthy",
            data["status"] == "healthy",
            f"version={data['version']}, groq={data['groq_configured']}, env={data.get('environment', 'unknown')}"
        )
        test(
            "Cache stats in health",
            "cache" in data,
            f"entries={data['cache'].get('entries', 0)}"
        )
        test(
            "API keys count in health",
            "api_keys_loaded" in data,
            f"keys={data.get('api_keys_loaded', 0)}"
        )
    except Exception as e:
        test("Server is healthy", False, str(e))
        print("⛔ Server not running! Start it with: python main.py")
        sys.exit(1)

    # ── Test 2: Agent Self-Discovery ──────────────────────────
    print("─── Test 2: Agent Self-Discovery ───")
    r = requests.get(f"{BASE}/api/skill", timeout=5)
    manifest = r.json()

    test(
        "Manifest has skill metadata",
        "skill" in manifest and manifest["skill"]["name"] == "doc-squeeze",
        f"name={manifest['skill']['name']}"
    )
    test(
        "Manifest has tools array",
        "tools" in manifest and len(manifest["tools"]) >= 4,
        f"tools: {[t['name'] for t in manifest['tools']]}"
    )

    # MCP Discovery
    r = requests.get(f"{BASE}/.well-known/mcp.json", timeout=5)
    mcp = r.json()
    test(
        "MCP discovery works",
        mcp.get("name") == "doc-squeeze" and len(mcp.get("tools", [])) >= 4,
        f"tools: {[t['name'] for t in mcp.get('tools', [])]}"
    )

    # OpenAI Plugin
    r = requests.get(f"{BASE}/.well-known/ai-plugin.json", timeout=5)
    plugin = r.json()
    test(
        "OpenAI plugin manifest works",
        plugin.get("name_for_model") == "doc_squeeze",
        f"schema={plugin.get('schema_version')}"
    )

    # ── Test 3: URL Validation (Security) ─────────────────────
    print("─── Test 3: URL Validation (Security) ───")

    r = requests.post(f"{BASE}/api/squeeze", json={
        "url": "http://localhost:8000/test"
    }, timeout=5)
    test("Blocks localhost URLs", r.status_code in [400, 422], f"status={r.status_code}")

    r = requests.post(f"{BASE}/api/squeeze", json={
        "url": "http://192.168.1.1/admin"
    }, timeout=5)
    test("Blocks private IPs", r.status_code in [400, 422], f"status={r.status_code}")

    r = requests.post(f"{BASE}/api/squeeze", json={
        "url": "http://169.254.169.254/latest/meta-data/"
    }, timeout=5)
    test("Blocks AWS metadata", r.status_code in [400, 422], f"status={r.status_code}")

    # Batch SSRF protection
    r = requests.post(f"{BASE}/api/batch", json={
        "urls": ["http://localhost:8000/test"]
    }, timeout=5)
    test("Blocks localhost in batch", r.status_code in [400, 422], f"status={r.status_code}")

    r = requests.post(f"{BASE}/api/batch", json={
        "urls": ["http://192.168.1.1/admin"]
    }, timeout=5)
    test("Blocks private IPs in batch", r.status_code in [400, 422], f"status={r.status_code}")

    # ── Test 4: Raw Doc Fetch ─────────────────────────────────
    print("─── Test 4: Raw Doc Fetch — No Focus ───")
    start = time.time()
    r = requests.post(f"{BASE}/api/squeeze", json={
        "url": "https://example.com"
    }, timeout=30)
    elapsed = round(time.time() - start, 2)

    if r.status_code == 200:
        data = r.json()
        test("Returns 200", True)
        test("Status is success", data["status"] == "success")
        test("Source is jina.ai", data["source"] == "jina.ai")
        test("Was NOT filtered", data["was_filtered"] is False)
        test("Has timing info", "timing" in data, f"total_ms={data['timing']['total_ms']}")
        test("Has cache flag", "cached" in data, f"cached={data['cached']}")
        test(
            "Content contains expected text",
            "example" in data["markdown"].lower(),
            f"chars={data['char_count']}, took {elapsed}s"
        )
    else:
        test("Returns 200", False, f"Got {r.status_code}: {r.text[:100]}")

    # ── Test 5: Response Headers ──────────────────────────────
    print("─── Test 5: Response Headers ───")
    test("X-Response-Time header", "X-Response-Time" in r.headers, r.headers.get("X-Response-Time", "missing"))
    test("X-Request-ID header", "X-Request-ID" in r.headers, r.headers.get("X-Request-ID", "missing"))
    test("X-API-Tier header", "X-API-Tier" in r.headers, r.headers.get("X-API-Tier", "missing"))
    test("X-Cache header", "X-Cache" in r.headers, r.headers.get("X-Cache", "missing"))

    # ── Test 6: Cache Hit ─────────────────────────────────────
    print("─── Test 6: Cache — Second Request Should Hit ───")
    start = time.time()
    r = requests.post(f"{BASE}/api/squeeze", json={
        "url": "https://example.com"
    }, timeout=30)
    elapsed = round(time.time() - start, 2)

    if r.status_code == 200:
        data = r.json()
        test("Cache HIT on repeat", data.get("cached") is True, f"took {elapsed}s")
        test("X-Cache is HIT", r.headers.get("X-Cache") == "HIT")
    else:
        test("Cache HIT on repeat", False, f"Got {r.status_code}")

    # ── Test 7: Cache Stats ───────────────────────────────────
    print("─── Test 7: Cache Stats ───")
    r = requests.get(f"{BASE}/api/cache/stats", timeout=5)
    stats = r.json()
    test("Cache stats endpoint works", r.status_code == 200)
    test("Has hit rate", "hit_rate_percent" in stats, f"hit_rate={stats.get('hit_rate_percent')}%")
    test("Has entries count", "entries" in stats, f"entries={stats.get('entries')}")

    # ── Test 8: API Key Creation ──────────────────────────────
    print("─── Test 8: API Key System ───")
    r = requests.post(f"{BASE}/api/keys/create", timeout=5)
    key_data = r.json()
    test("Key creation works", r.status_code == 200 and "api_key" in key_data)
    api_key = key_data.get("api_key", "")
    test("Key has ds_ prefix", api_key.startswith("ds_"), f"key={api_key[:12]}...")
    test("Returns tier info", key_data.get("tier") == "dev")
    test("Returns features list", "features" in key_data, f"features={key_data.get('features')}")

    # Test authenticated request gets higher tier
    r = requests.post(f"{BASE}/api/squeeze",
        json={"url": "https://example.com"},
        headers={"X-API-Key": api_key},
        timeout=30
    )
    test(
        "Authenticated tier = dev",
        r.headers.get("X-API-Tier") == "dev",
        f"tier={r.headers.get('X-API-Tier')}"
    )

    # ── Test 9: Cache Clear Requires Auth ─────────────────────
    print("─── Test 9: Cache Clear Auth ───")
    r = requests.delete(f"{BASE}/api/cache", timeout=5)
    test("Cache clear blocked without key", r.status_code == 403)

    r = requests.delete(f"{BASE}/api/cache",
        headers={"X-API-Key": api_key},
        timeout=5
    )
    test("Cache clear works with key", r.status_code == 200, f"response={r.json()}")

    # ── Test 10: Focused Fetch (Groq) ─────────────────────────
    print("─── Test 10: Focused Fetch — With Groq ───")
    start = time.time()
    r = requests.post(f"{BASE}/api/squeeze", json={
        "url": "https://docs.python.org/3/library/json.html",
        "focus": "json.dumps function parameters"
    }, timeout=60)
    elapsed = round(time.time() - start, 2)

    if r.status_code == 200:
        data = r.json()
        test("Filtered returns 200", True)
        test("Was filtered by Groq", data["was_filtered"] is True)
        test("Has model_used", data.get("model_used") is not None, f"model={data.get('model_used')}")
        test(
            "Mentions 'dumps'",
            "dumps" in data["markdown"].lower(),
            f"chars={data['char_count']}, took {elapsed}s"
        )
    else:
        test("Filtered returns 200", False, f"Got {r.status_code}")

    # ── Test 11: Structured Extraction ────────────────────────
    print("─── Test 11: Structured Extraction ───")
    start = time.time()
    r = requests.post(f"{BASE}/api/extract", json={
        "url": "https://example.com",
        "schema_definition": {
            "title": "str",
            "description": "str",
            "links": ["str"]
        }
    }, timeout=60)
    elapsed = round(time.time() - start, 2)

    if r.status_code == 200:
        data = r.json()
        test("Extract returns 200", True)
        test("Has extracted data", "data" in data and isinstance(data["data"], dict))
        test("Has confidence score", 0 <= data.get("confidence", -1) <= 1, f"confidence={data.get('confidence')}")
        test("Has timing", "timing" in data, f"total_ms={data['timing']['total_ms']}")
    else:
        test("Extract returns 200", False, f"Got {r.status_code}: {r.text[:100]}")

    # ── Test 12: Deep Search ──────────────────────────────────
    print("─── Test 12: Deep Search ───")
    start = time.time()
    r = requests.post(f"{BASE}/api/search", json={
        "url": "https://docs.python.org/3/library/json.html",
        "query": "How to serialize custom objects?",
        "max_results": 3
    }, timeout=60)
    elapsed = round(time.time() - start, 2)

    if r.status_code == 200:
        data = r.json()
        test("Search returns 200", True)
        test("Has results", len(data.get("results", [])) > 0, f"found {len(data.get('results', []))} results")
        if data.get("results"):
            first = data["results"][0]
            test("Results have section", "section" in first)
            test("Results have content", "content" in first)
            test("Results have relevance", 0 <= first.get("relevance", -1) <= 1, f"relevance={first.get('relevance')}")
    else:
        test("Search returns 200", False, f"Got {r.status_code}: {r.text[:100]}")

    # ── Test 13: Batch Processing ─────────────────────────────
    print("─── Test 13: Batch Processing ───")
    start = time.time()
    r = requests.post(f"{BASE}/api/batch", json={
        "urls": ["https://example.com", "https://httpbin.org/html"]
    }, timeout=60)
    elapsed = round(time.time() - start, 2)

    if r.status_code == 200:
        data = r.json()
        test("Batch returns 200", True)
        test("Has results array", len(data.get("results", [])) == 2, f"got {len(data.get('results', []))} results")
        test("Has total_urls", data.get("total_urls") == 2)
        test("Has success count", data.get("successful", 0) > 0, f"successful={data.get('successful')}")
        test("Has timing", "timing" in data, f"total_ms={data['timing']['total_ms']}")
    else:
        test("Batch returns 200", False, f"Got {r.status_code}: {r.text[:100]}")

    # ── Test 14: Edge Cases ───────────────────────────────────
    print("─── Test 14: Edge Cases ───")

    r = requests.post(f"{BASE}/api/squeeze", json={"focus": "something"}, timeout=5)
    test("Missing URL returns 422", r.status_code == 422)

    r = requests.post(f"{BASE}/api/squeeze", json={"url": "not-a-url"}, timeout=5)
    test("Invalid URL returns 422", r.status_code == 422)

    r = requests.post(f"{BASE}/api/batch", json={"urls": []}, timeout=5)
    test("Empty batch returns 422", r.status_code == 422)

    r = requests.post(f"{BASE}/api/search", json={
        "url": "https://example.com", "query": "test", "max_results": 20
    }, timeout=5)
    test("max_results>10 returns 422", r.status_code == 422)

    # ── Test 15: Error Response Format ────────────────────────
    print("─── Test 15: Error Response Format ───")
    r = requests.post(f"{BASE}/api/squeeze", json={
        "url": "http://localhost/test"
    }, timeout=5)

    if r.status_code != 200:
        try:
            error_data = r.json()
            test(
                "Error has standardized format",
                "error" in error_data or "detail" in error_data,
                f"keys={list(error_data.keys())}"
            )
        except Exception:
            test("Error has standardized format", False, "Not JSON response")

    # ── Summary ───────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print("=" * 60)
    print(f"📊  RESULTS: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print("  — ALL TESTS PASSED! 🎉")
    print("=" * 60)
    print()

    if failed:
        print("Failed tests:")
        for name, p in results:
            if not p:
                print(f"  {FAIL} {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
