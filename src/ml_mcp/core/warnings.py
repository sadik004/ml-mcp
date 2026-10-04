"""Request-scoped structured audit warnings collector (Phase P4).

Guarantees zero silent fallbacks by tracking surrogate models, numerical warnings, and heuristic triggers.
"""
from __future__ import annotations

from typing import List, Union


class WarningsCollector:
    """Collects and formats structured audit warnings during tool execution."""

    def __init__(self) -> None:
        self._warnings: List[str] = []

    def add(self, code: str, message: str) -> None:
        """Add a structured warning with code and explanatory message."""
        formatted = f"[{code}] {message}"
        if formatted not in self._warnings:
            self._warnings.append(formatted)

    def as_list(self) -> List[str]:
        """Return the collected list of warning strings."""
        return list(self._warnings)

    def has_warnings(self) -> bool:
        """Check if any warnings have been recorded."""
        return len(self._warnings) > 0

    def merge(self, other: Union[WarningsCollector, List[str]]) -> None:
        """Merge warnings from another collector or string list."""
        items = other.as_list() if isinstance(other, WarningsCollector) else other
        for item in items:
            if item not in self._warnings:
                self._warnings.append(item)

    def __repr__(self) -> str:
        return f"<WarningsCollector count={len(self._warnings)} warnings={self._warnings}>"
