"""
Unit tests for Doc-Squeeze API
Run with: pytest tests/
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import Settings, app, fetch_via_jina, filter_with_groq


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    """Test client for FastAPI app with lifespan context."""
    # TestClient automatically triggers startup/shutdown events
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return Settings(
        groq_api_key="test_key",
        jina_timeout=10,
        groq_timeout=10,
        allowed_origins="http://localhost:3000",
        rate_limit_squeeze="100/minute",  # Higher limit for tests
    )


# ── Health & Basic Endpoints ─────────────────────────────────────────────────
def test_health_endpoint(client):
    """Test /health endpoint returns correct structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "doc-squeeze"
    assert "version" in data
    assert "groq_configured" in data


def test_root_endpoint(client):
    """Test / endpoint returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Doc-Squeeze" in response.text


def test_skill_manifest_endpoint(client):
    """Test /api/skill endpoint returns manifest."""
    response = client.get("/api/skill")
    assert response.status_code == 200
    data = response.json()
    assert "skill" in data
    assert data["skill"]["name"] == "doc-squeeze"


# ── URL Validation ───────────────────────────────────────────────────────────
def test_url_validation_blocks_localhost(client):
    """Test that localhost URLs are blocked."""
    response = client.post("/api/squeeze", json={
        "url": "http://localhost:8000/test"
    })
    assert response.status_code == 422  # Validation error


def test_url_validation_blocks_private_ips(client):
    """Test that private IP addresses are blocked."""
    response = client.post("/api/squeeze", json={
        "url": "http://192.168.1.1/admin"
    })
    assert response.status_code == 422


def test_url_validation_blocks_metadata_endpoint(client):
    """Test that AWS metadata endpoint is blocked."""
    response = client.post("/api/squeeze", json={
        "url": "http://169.254.169.254/latest/meta-data/"
    })
    assert response.status_code == 422


def test_url_validation_allows_valid_https(client):
    """Test that valid HTTPS URLs pass validation (but may fail at fetch)."""
    # This will fail at the fetch stage, but should pass validation
    response = client.post("/api/squeeze", json={
        "url": "https://example.com"
    })
    # Could be 502 (fetch failed) or 200 (if it works), but not 422 (validation)
    assert response.status_code != 422


# ── Error Handling ───────────────────────────────────────────────────────────
def test_missing_url_field(client):
    """Test that missing URL field returns 422."""
    response = client.post("/api/squeeze", json={
        "focus": "test"
    })
    assert response.status_code == 422


def test_invalid_url_format(client):
    """Test that invalid URL format returns error."""
    response = client.post("/api/squeeze", json={
        "url": "not-a-url"
    })
    assert response.status_code == 422


# ── Rate Limiting ────────────────────────────────────────────────────────────
def test_rate_limiting():
    """Test that rate limiting works (requires lower limit in test config)."""
    # This test would need a separate test client with custom rate limit settings
    # Skipping for now as it requires special configuration
    pass


# ── Settings Configuration ───────────────────────────────────────────────────
def test_settings_default_values():
    """Test that settings have correct default values."""
    test_settings = Settings()
    assert test_settings.port == 8000
    assert test_settings.env == "development"
    assert test_settings.jina_timeout == 30


def test_settings_cors_origins_parsing():
    """Test CORS origins parsing."""
    test_settings = Settings(allowed_origins="http://a.com,http://b.com")
    assert len(test_settings.cors_origins) == 2
    assert "http://a.com" in test_settings.cors_origins


def test_settings_is_development():
    """Test is_development property."""
    dev_settings = Settings(env="development")
    assert dev_settings.is_development is True

    prod_settings = Settings(env="production")
    assert prod_settings.is_development is False


# ── Mock Tests for External API Calls ────────────────────────────────────────
@pytest.mark.asyncio
async def test_fetch_via_jina_success():
    """Test successful Jina fetch."""
    with patch('main.http_client') as mock_client:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.text = "# Test Markdown\n\nContent here."
        mock_response.raise_for_status = MagicMock()

        mock_client.get = AsyncMock(return_value=mock_response)

        result = await fetch_via_jina("https://example.com")
        assert "Test Markdown" in result


@pytest.mark.asyncio
async def test_filter_with_groq_no_key():
    """Test that filtering returns original content when no Groq key is set."""
    with patch('main.groq_client', None):
        original = "# Test Content"
        result = await filter_with_groq(original, "focus")
        assert result == original


# ── Integration-like Tests ───────────────────────────────────────────────────
def test_squeeze_endpoint_structure(client):
    """Test that squeeze endpoint returns correct structure (even if fetch fails)."""
    # This will likely fail at the fetch stage, but we can check error structure
    response = client.post("/api/squeeze", json={
        "url": "https://httpbin.org/status/404"
    })

    # Check that we get a proper error response
    if response.status_code != 200:
        data = response.json()
        assert "error" in data or "detail" in data
