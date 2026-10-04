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
    error: str | Exception | None = None,
    message_or_tool: str = "",
    details: Optional[Dict[str, Any] | List[str]] = None,
    candidates: Optional[List[str]] = None,
    exception: Optional[Exception] = None,
    retryable: bool = True,
    *,
    error_type: Optional[str] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """Formats an exception or validation error into a structured, self-healing JSON envelope.

    Supports both:
    1. format_error_envelope(exc, "tool_name", candidates=["col1", "col2"])
    2. format_error_envelope(error_type="ValidationError", message="msg", details={...})
    """
    tool_name: Optional[str] = None
    resolved_exception: Optional[Exception] = None
    resolved_candidates: Optional[List[str]] = None
    resolved_details: Dict[str, Any] = {}

    if isinstance(error, Exception):
        exc_instance = error
        err_type = exc_instance.__class__.__name__
        tool_name = message_or_tool
        msg = f"{tool_name}: {exc_instance}" if tool_name else str(exc_instance)
        resolved_exception = exc_instance
        resolved_details = {"tool": tool_name}
        if isinstance(details, dict):
            resolved_details.update(details)
        resolved_candidates = details if isinstance(details, list) else candidates
    else:
        err_type = error_type if error_type is not None else (str(error) if error is not None else "UnknownError")
        msg = message if message is not None else message_or_tool
        resolved_exception = exception
        resolved_details = details if isinstance(details, dict) else {}
        resolved_candidates = candidates

    suggestions: List[str] = []
    if resolved_candidates and "attempted_column" in resolved_details:
        suggestions = suggest_close_matches(str(resolved_details["attempted_column"]), resolved_candidates)

    err_code = getattr(resolved_exception, "error_code", "INTERNAL_ERROR") if resolved_exception is not None else "INTERNAL_ERROR"
    if resolved_exception is not None:
        exc_details = getattr(resolved_exception, "details", None)
        if isinstance(exc_details, dict):
            resolved_details.update(exc_details)


    step_name = tool_name or (resolved_details.get("tool") if isinstance(resolved_details, dict) else None)
    remed = suggestions if suggestions else ["Verify input parameters and column names."]

    envelope: Dict[str, Any] = {
        "status": "error",
        "error_type": err_type,
        "error_code": err_code,
        "message": msg,
        "error_message": msg,
        "failed_step": step_name,
        "retryable": retryable,
        "details": resolved_details,
        "inputs_provided": resolved_details,
        "suggestions": suggestions,
        "remediation_suggestions": remed,
    }

    if resolved_exception is not None:
        envelope["exception_class"] = resolved_exception.__class__.__name__

    return sanitize_for_json(envelope)
