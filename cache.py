"""
Smart caching layer for Doc-Squeeze.

TTL-based in-memory cache that dramatically reduces response times for
repeated requests. Same docs page requested by 100 agents? Sub-50ms
from cache instead of 2-5s round-trip to Jina.

Thread-safe, zero dependencies (stdlib only).
"""

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheEntry:
    """A single cached item with metadata."""
    value: Any
    created_at: float
    ttl: int  # seconds
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl

    @property
    def age_seconds(self) -> int:
        return int(time.time() - self.created_at)


class SmartCache:
    """
    TTL-based in-memory cache with stats tracking.

    Usage:
        cache = SmartCache(default_ttl=900)  # 15 min default

        # Try cache first
        result = cache.get("key")
        if result is None:
            result = expensive_fetch()
            cache.set("key", result)
    """

    def __init__(self, default_ttl: int = 900, max_entries: int = 500):
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl
        self.max_entries = max_entries

        # Stats
        self._total_hits = 0
        self._total_misses = 0
        self._total_sets = 0
        self._created_at = time.time()

    @staticmethod
    def make_key(*parts: str) -> str:
        """Create a deterministic cache key from parts."""
        raw = "|".join(str(p) for p in parts if p)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache. Returns None on miss or expiry."""
        with self._lock:
            entry = self._store.get(key)

            if entry is None:
                self._total_misses += 1
                return None

            if entry.is_expired:
                del self._store[key]
                self._total_misses += 1
                return None

            entry.hits += 1
            self._total_hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in cache with optional custom TTL."""
        with self._lock:
            # Evict expired entries if we're at capacity
            if len(self._store) >= self.max_entries:
                self._evict_expired()

            # If still at capacity, evict oldest
            if len(self._store) >= self.max_entries:
                self._evict_oldest()

            self._store[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl or self.default_ttl,
            )
            self._total_sets += 1

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries. Returns count of cleared entries."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total_requests = self._total_hits + self._total_misses
            hit_rate = (
                round(self._total_hits / total_requests * 100, 1)
                if total_requests > 0 else 0
            )

            # Count active (non-expired) entries
            active = sum(1 for e in self._store.values() if not e.is_expired)

            return {
                "entries": len(self._store),
                "active_entries": active,
                "max_entries": self.max_entries,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "total_sets": self._total_sets,
                "hit_rate_percent": hit_rate,
                "default_ttl_seconds": self.default_ttl,
                "uptime_seconds": int(time.time() - self._created_at),
            }

    def _evict_expired(self) -> None:
        """Remove all expired entries (must hold lock)."""
        expired_keys = [k for k, v in self._store.items() if v.is_expired]
        for k in expired_keys:
            del self._store[k]

    def _evict_oldest(self) -> None:
        """Remove the oldest entry (must hold lock)."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
        del self._store[oldest_key]


# ── Global cache instance ────────────────────────────────────────────────────
content_cache = SmartCache(default_ttl=900, max_entries=500)
