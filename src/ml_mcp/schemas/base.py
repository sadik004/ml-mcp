"""Unified BaseDTO for all ml_mcp schemas with token shielding and strict validation."""
from __future__ import annotations

from typing import Any, Dict, Literal
from pydantic import BaseModel, ConfigDict, Field


class BaseDTO(BaseModel):
    """Root DTO enforcing strict field contracts and token-guard view filtering."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    view: Literal["compact", "detailed"] = Field(
        default="compact",
        description="Token guard filter mode: compact (<300 tokens) vs detailed full dump",
    )

    def to_compact(self) -> Dict[str, Any]:
        """Returns a high-density, low-token summary dictionary for LLM context preservation."""
        dump = self.model_dump()
        dump.pop("view", None)
        return dump
