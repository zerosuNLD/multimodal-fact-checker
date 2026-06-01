"""Shared event emitter for streaming agent progress to SSE."""

import asyncio
import json
from contextvars import ContextVar
from typing import Dict

# ContextVar to access the current session's SSE queue from nodes
_current_session_id: ContextVar[str] = ContextVar("current_session_id", default="")

# Global dict: session_id → asyncio.Queue
_session_queues: Dict[str, asyncio.Queue] = {}


def register_session(session_id: str, queue: asyncio.Queue):
    """Register a new session's SSE queue."""
    _session_queues[session_id] = queue


def unregister_session(session_id: str):
    """Remove a session's SSE queue."""
    _session_queues.pop(session_id, None)


def set_current_session(session_id: str):
    """Set the current session in the contextvar."""
    _current_session_id.set(session_id)


async def emit_event(event: str, data: dict):
    """Push an SSE event to the current session's queue."""
    sid = _current_session_id.get()
    if not sid:
        return
    q = _session_queues.get(sid)
    if q:
        await q.put({"event": event, "data": data})
