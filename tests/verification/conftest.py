"""Pytest configuration for verification tests. Enforces zero skip/xfail integrity."""
from pathlib import Path
from typing import Any, List
import pytest


def pytest_collection_modifyitems(session: Any, config: Any, items: List[Any]) -> None:
    """Prohibit skip, skipif, and xfail markers in all verification tests."""
    verification_dir = Path(__file__).parent.resolve()
    for item in items:
        try:
            item_path = Path(str(item.fspath)).resolve()
        except Exception:
            continue

        if verification_dir in item_path.parents or item_path == verification_dir:
            for marker in item.iter_markers():
                if marker.name in ("skip", "skipif", "xfail"):
                    raise pytest.UsageError(
                        f"BANNED TEST SUPPRESSION: '{item.nodeid}' uses marker '@pytest.mark.{marker.name}'. "
                        "Tests under tests/verification/ are strictly prohibited from using skip, skipif, or xfail (Rule T2)."
                    )
