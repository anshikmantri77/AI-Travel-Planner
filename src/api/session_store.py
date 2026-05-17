"""In-memory session store with TTL-based expiry.

Stores LangGraph thread IDs and metadata for each planning session.
Thread-safe via ``asyncio.Lock``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

from src.config import get_settings


class SessionStore:
    """Async-safe in-memory session store."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> str:
        """Create a new session and return its ID."""
        session_id = uuid4().hex[:12]
        async with self._lock:
            self._store[session_id] = {
                "thread_id": session_id,  # use same id for LangGraph thread
                "created_at": time.time(),
                "status": "created",
                "workflow_stage": "created",
            }
        return session_id

    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve session metadata, or None if not found / expired."""
        async with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            ttl = get_settings().SESSION_TTL_SECONDS
            if time.time() - entry["created_at"] > ttl:
                del self._store[session_id]
                return None
            return dict(entry)

    async def update(self, session_id: str, **kwargs: Any) -> bool:
        """Update session metadata fields.  Returns False if not found."""
        async with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return False
            entry.update(kwargs)
            return True

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._store.pop(session_id, None) is not None

    async def cleanup_expired(self) -> int:
        """Remove expired sessions.  Returns count of removed entries."""
        ttl = get_settings().SESSION_TTL_SECONDS
        now = time.time()
        removed = 0
        async with self._lock:
            expired = [
                sid for sid, entry in self._store.items()
                if now - entry["created_at"] > ttl
            ]
            for sid in expired:
                del self._store[sid]
                removed += 1
        return removed

    @property
    def size(self) -> int:
        return len(self._store)
