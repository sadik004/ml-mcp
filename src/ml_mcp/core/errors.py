"""Typed exception hierarchy for ml-mcp (Phase P4).

Enforces predictable, structured error reporting across all MCP tools, routers, and engines.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class MLMCPError(Exception):
    """Base class for all domain exceptions in ml-mcp."""

    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CertificationRefused(MLMCPError):
    """Raised when an independent safety certificate cannot be honestly granted."""

    error_code: str = "CERTIFICATION_REFUSED"


class InsufficientSamples(MLMCPError):
    """Raised when sample size is too small for statistical validity (e.g. Wilson CI, CRC, conformal)."""

    error_code: str = "INSUFFICIENT_SAMPLES"


class DependencyMissing(MLMCPError):
    """Raised when an explicit dependency is required in strict mode but missing from runtime."""

    error_code: str = "DEPENDENCY_MISSING"


class CorruptArtifact(MLMCPError):
    """Raised when a persisted checkpoint, pipeline, or OOF prediction file cannot be safely deserialized."""

    error_code: str = "CORRUPT_ARTIFACT"


class InvalidInput(MLMCPError):
    """Raised when client input data, schemas, or target columns fail invariant validation."""

    error_code: str = "INVALID_INPUT"
