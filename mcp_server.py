"""
Doc-Squeeze MCP Server — JSON-RPC 2.0 over stdio transport.

Exposes all Doc-Squeeze tools as native MCP tools, compatible with:
- Claude Desktop
- Cursor
- Windsurf
- Any MCP-compatible AI agent

Tools:
- squeeze_url: Fetch URL as clean markdown
- extract_structured: Extract JSON data using a schema
- search_docs: Search within a page for answers
- batch_squeeze: Fetch multiple URLs in parallel
- health_check: Check service health

Usage:
    python mcp_server.py

Configuration for Claude Desktop (claude_desktop_config.json):
    {
        "mcpServers": {
            "doc-squeeze": {
                "command": "python",
                "args": ["/path/to/mcp_server.py"]
            }
        }
    }
"""

import json
import os
import sys

import httpx

# Default base URL — override with DOC_SQUEEZE_URL env var
BASE_URL = os.environ.get("DOC_SQUEEZE_URL", "http://localhost:8000")


# ── MCP Protocol Helpers ─────────────────────────────────────────────────────
def send_response(id_val, result):
    """Send a JSON-RPC 2.0 response."""
    response = {"jsonrpc": "2.0", "id": id_val, "result": result}
    msg = json.dumps(response)
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def send_error(id_val, code, message, data=None):
    """Send a JSON-RPC 2.0 error response."""
    error = {"code": code, "message": message}
    if data:
        error["data"] = data
    response = {"jsonrpc": "2.0", "id": id_val, "error": error}
    msg = json.dumps(response)
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ── Tool Definitions ─────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "squeeze_url",
        "description": (
            "Fetch a documentation URL and return its content as clean markdown. "
            "Optionally filter to a specific topic using an LLM. "
            "Use this when you need to read external docs, API references, or tutorials."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The full URL of the documentation page to read.",
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "Optional topic to filter the documentation to. "
                        "Examples: 'authentication', 'error handling', 'Python SDK setup'."
                    ),
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "extract_structured",
        "description": (
            "Extract structured JSON data from a URL using a schema definition. "
            "Instead of raw markdown, get typed, parseable data. "
            "Send a URL + a JSON schema describing what you want."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "URL to extract data from.",
                },
                "schema_definition": {
                    "type": "object",
                    "description": "JSON schema defining the structure to extract.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Optional extraction guidance.",
                },
            },
            "required": ["url", "schema_definition"],
        },
    },
    {
        "name": "search_docs",
        "description": (
            "Search within a documentation page for answers to a specific query. "
            "Returns the most relevant sections with relevance scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "URL of the page to search within.",
                },
                "query": {
                    "type": "string",
                    "description": "The question or topic to search for.",
                },
                "max_results": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum number of results (1-10).",
                },
            },
            "required": ["url", "query"],
        },
    },
    {
        "name": "batch_squeeze",
        "description": (
            "Fetch multiple documentation URLs in parallel. "
            "Up to 10 URLs per request. Optionally filter all to a focus topic."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string", "format": "uri"},
                    "description": "List of URLs to fetch (max 10).",
                },
                "focus": {
                    "type": "string",
                    "description": "Optional topic filter applied to all URLs.",
                },
            },
            "required": ["urls"],
        },
    },
    {
        "name": "health_check",
        "description": "Check if the Doc-Squeeze service is running and healthy.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Tool Execution ───────────────────────────────────────────────────────────
def call_squeeze(args):
    """Execute squeeze_url tool."""
    url = args.get("url")
    focus = args.get("focus")

    if not url:
        return {"error": "url parameter is required"}

    payload = {"url": url}
    if focus:
        payload["focus"] = focus

    try:
        resp = httpx.post(f"{BASE_URL}/api/squeeze", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": str(e)}


def call_extract(args):
    """Execute extract_structured tool."""
    url = args.get("url")
    schema_def = args.get("schema_definition")

    if not url or not schema_def:
        return {"error": "url and schema_definition are required"}

    payload = {"url": url, "schema_definition": schema_def}
    instructions = args.get("instructions")
    if instructions:
        payload["instructions"] = instructions

    try:
        resp = httpx.post(f"{BASE_URL}/api/extract", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": str(e)}


def call_search(args):
    """Execute search_docs tool."""
    url = args.get("url")
    query = args.get("query")

    if not url or not query:
        return {"error": "url and query are required"}

    payload = {"url": url, "query": query}
    max_results = args.get("max_results")
    if max_results:
        payload["max_results"] = max_results

    try:
        resp = httpx.post(f"{BASE_URL}/api/search", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": str(e)}


def call_batch(args):
    """Execute batch_squeeze tool."""
    urls = args.get("urls")

    if not urls:
        return {"error": "urls parameter is required"}

    payload = {"urls": urls}
    focus = args.get("focus")
    if focus:
        payload["focus"] = focus

    try:
        resp = httpx.post(f"{BASE_URL}/api/batch", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": str(e)}


def call_health():
    """Execute health_check tool."""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


TOOL_HANDLERS = {
    "squeeze_url": call_squeeze,
    "extract_structured": call_extract,
    "search_docs": call_search,
    "batch_squeeze": call_batch,
    "health_check": lambda _: call_health(),
}


# ── JSON-RPC 2.0 Handler ────────────────────────────────────────────────────
def handle_request(request):
    """Process a single JSON-RPC 2.0 request."""
    method = request.get("method", "")
    id_val = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        send_response(id_val, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": "doc-squeeze",
                "version": "1.0.0",
            },
        })

    elif method == "notifications/initialized":
        pass  # Client acknowledged, no response needed

    elif method == "tools/list":
        send_response(id_val, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            send_error(id_val, -32601, f"Unknown tool: {tool_name}")
            return

        result = handler(tool_args)
        send_response(id_val, {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
        })

    elif method == "ping":
        send_response(id_val, {})

    else:
        send_error(id_val, -32601, f"Method not found: {method}")


# ── Main Loop ────────────────────────────────────────────────────────────────
def main():
    """Run the MCP server — reads JSON-RPC messages from stdin."""
    sys.stderr.write("Doc-Squeeze MCP server v1.0.0 started\n")
    sys.stderr.write(f"Backend URL: {BASE_URL}\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            handle_request(request)
        except json.JSONDecodeError:
            send_error(None, -32700, "Parse error: invalid JSON")
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.stderr.flush()
            send_error(None, -32603, f"Internal error: {e}")


if __name__ == "__main__":
    main()
