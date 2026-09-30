"""Standardized self-healing error envelope formatter with fuzzy matching suggestions.

Ensures that errors returned to LLM clients (Antigravity/Claude Code) are informative,
token-efficient, and actionable without flooding the context window with raw tracebacks.
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional
from ml_mcp.engine.json_sanitizer import sanitize_for_json


def suggest_close_matches(
    query: str, candidates: List[str], n: int = 3, cutoff: float = 0.5
) -> List[str]:
    """Finds closest fuzzy matches for a misspelled feature or parameter name."""
    if not query or not candidates:
        return []
    return difflib.get_close_matches(query, candidates, n=n, cutoff=cutoff)


def format_error_envelope(
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    candidates: Optional[List[str]] = None,
    exception: Optional[Exception] = None,
    retryable: bool = True,
) -> Dict[str, Any]:
    """Formats an exception or validation error into a structured, self-healing JSON envelope.

    Args:
        error_type: Name or category of error (e.g. ValidationError, DataLeakageError).
        message: Concise description of the issue.
        details: Optional key-value pairs providing context (attempted feature, valid ranges).
        candidates: Optional pool of valid column or model names to run fuzzy matching on.
        exception: The underlying exception instance if caught.
        retryable: Whether the agent can correct its parameters and retry immediately.

    Returns:
        JSON-sanitized structured error payload.
    """
    suggestions: List[str] = []
    if candidates and details and "attempted_column" in details:
        suggestions = suggest_close_matches(details["attempted_column"], candidates)

    envelope: Dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "retryable": retryable,
        "details": details or {},
        "suggestions": suggestions,
    }

    if exception is not None:
        envelope["exception_class"] = exception.__class__.__name__

    return sanitize_for_json(envelope)
