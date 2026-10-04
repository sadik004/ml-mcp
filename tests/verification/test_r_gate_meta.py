"""RED Verification Suite: Quality Gate Blind Spots & AST Detection (Gate Hardening).

Enforces Rule T3 (independent AST inspection) and tests the gates on fixtures.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

import numpy as np
import pytest



def _check_silent_except_ast(code: str) -> List[str]:
    """Flag broad except that returns a value without warning/raising."""
    tree = ast.parse(code)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                # Catch broad exception
                is_broad = h.type is None or (
                    isinstance(h.type, ast.Name) and h.type.id in ("Exception", "BaseException")
                )
                if is_broad:
                    body_code = ast.unparse(h)
                    has_warn_or_raise = (
                        "warning" in body_code.lower()
                        or "logger" in body_code.lower()
                        or any(isinstance(n, ast.Raise) for n in h.body)
                    )
                    if not has_warn_or_raise:
                        violations.append(f"L{h.lineno}: silent except without warning/raise")
    return violations


def _check_hardcoded_seeds_ast(code: str) -> List[str]:
    """Flag integer literals in seed / random_state parameters and defaults."""
    tree = ast.parse(code)
    violations = []
    for node in ast.walk(tree):
        # Check function def default args
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg, default in zip(reversed(node.args.args), reversed(node.args.defaults)):
                if arg.arg in ("seed", "random_state", "random_seed"):
                    if isinstance(default, ast.Constant) and isinstance(default.value, int):
                        violations.append(f"L{node.lineno}: Hardcoded default {arg.arg}={default.value}")
        # Check call keywords
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("seed", "random_state", "random_seed"):
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                        violations.append(f"L{node.lineno}: Hardcoded call keyword {kw.arg}={kw.value.value}")
    return violations


def test_g1_flags_broad_except_without_warning() -> None:
    """Gate G1 must flag broad except returning fallback value with no warnings."""
    bad_code = """
try:
    x = 1 / 0
except Exception:
    oof_score = float(stacking_model.score(X, y))
"""
    violations = _check_silent_except_ast(bad_code)
    assert len(violations) >= 1, "Failed to flag silent except returning in-sample score!"


def test_g1_allows_except_with_warning_append() -> None:
    """Gate G1 must allow except that registers a warning and does not invent data."""
    good_code = """
try:
    x = 1 / 0
except Exception as e:
    warnings.append(f"Calculation failed: {e}")
    oof_score = None
"""
    violations = _check_silent_except_ast(good_code)
    assert len(violations) == 0, f"False positive on logged exception: {violations}"


def test_seed_check_flags_all_hardcoded_forms() -> None:
    """Seed check must flag integer literals in random_state/seed defaults and calls."""
    bad_code = """
def split(data, random_state: int = 42):
    model = CatBoostRegressor(random_seed=42)
    sampler = TPESampler(seed=42)
"""
    violations = _check_hardcoded_seeds_ast(bad_code)
    assert len(violations) == 3, f"Expected 3 seed violations, found {len(violations)}: {violations}"


def test_current_src_contains_zero_violations_gate() -> None:
    """RED Gate Check: Current src/ml_mcp still contains 16 hardcoded seeds and C2 fallback."""
    src_dir = Path("src/ml_mcp")
    all_seed_violations = []
    for py_file in src_dir.rglob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        v = _check_hardcoded_seeds_ast(code)
        if v:
            all_seed_violations.extend([f"{py_file.name}:{item}" for item in v])

    # Currently there are 16 hardcoded seeds in src/ml_mcp (config.py defaults, etc.)
    # This assertion verifies that the suite flags them until P7 remediates them.
    assert len(all_seed_violations) == 0, f"Found {len(all_seed_violations)} hardcoded seeds in src/ml_mcp: {all_seed_violations[:3]}..."
