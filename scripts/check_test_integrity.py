"""G0: Test Integrity and Anti-Gaming Audit Gate (AST-based).

Enforces Rules T1-T11 across tests/verification/ and src/.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Tolerances & constants
MAX_UNBOUNDED_APPROX = 1e-9


class IntegrityViolation:
    """Represents a test integrity violation."""

    def __init__(self, file: str, line: int, rule: str, message: str) -> None:
        self.file = file
        self.line = line
        self.rule = rule
        self.message = message

    def __str__(self) -> str:
        return f"{self.file}:{self.line} [{self.rule}] {self.message}"


class TestIntegrityAuditor:
    """Audits test files and source files against Test-Integrity Contract."""

    __test__ = False

    def __init__(
        self,
        root_dir: Path,
        verification_dir: Optional[Path] = None,
        src_dir: Optional[Path] = None,
        lock_file: Optional[Path] = None,
        red_evidence_file: Optional[Path] = None,
        lock_changes_file: Optional[Path] = None,
        mode: str = "enforce",
        require_colab: bool = False,
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.verification_dir = (verification_dir or (self.root_dir / "tests" / "verification")).resolve()
        self.src_dir = (src_dir or (self.root_dir / "src")).resolve()
        self.lock_file = (lock_file or (self.verification_dir / "LOCK.json")).resolve()
        self.red_evidence_file = (
            red_evidence_file or (self.root_dir / "docs" / "plans" / "remediation_v3" / "RED_EVIDENCE.md")
        ).resolve()
        self.lock_changes_file = (
            lock_changes_file or (self.root_dir / "docs" / "plans" / "remediation_v3" / "LOCK_CHANGES.md")
        ).resolve()
        self.mode = mode
        self.require_colab = require_colab
        self.violations: List[IntegrityViolation] = []

    def log(self, file: Path, line: int, rule: str, msg: str) -> None:
        try:
            rel = file.relative_to(self.root_dir)
        except Exception:
            rel = file
        self.violations.append(IntegrityViolation(str(rel), line, rule, msg))

    def audit_all(self) -> List[IntegrityViolation]:
        self.violations.clear()
        self.check_t7_src_test_detection()

        test_files = list(self.verification_dir.glob("test_r_*.py"))
        # Also check subdirectories like heavy/
        test_files.extend(list(self.verification_dir.glob("**/test_r_*.py")))
        test_files = sorted(list({f.resolve() for f in test_files}))

        if not test_files and self.mode == "bootstrap":
            return self.violations

        self.check_t1_and_t9_lock(test_files)
        self.check_t5_red_evidence(test_files)

        for tf in test_files:
            self.audit_test_file(tf)

        if self.require_colab:
            self.check_t11_colab()

        return self.violations

    def check_t7_src_test_detection(self) -> None:
        """T7: Production code in src/ must not detect test runners."""
        if not self.src_dir.exists():
            return
        banned_patterns = [
            (re.compile(r'PYTEST_CURRENT_TEST'), "Usage of PYTEST_CURRENT_TEST environment variable"),
            (re.compile(r'sys\.modules\[["\']pytest["\']\]'), "Checking sys.modules for pytest"),
            (re.compile(r'\bimport\s+pytest\b'), "Importing pytest in production source code"),
            (re.compile(r'\bfrom\s+pytest\s+import\b'), "Importing from pytest in production source code"),
            (re.compile(r'\bimport\s+unittest\b'), "Importing unittest in production source code"),
        ]

        for py_path in self.src_dir.rglob("*.py"):
            try:
                lines = py_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, 1):
                # ignore comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pat, msg in banned_patterns:
                    if pat.search(line):
                        self.log(py_path, i, "T7", msg)

    def check_t1_and_t9_lock(self, test_files: List[Path]) -> None:
        """T1: SHA-256 test lock & T9: No test deletion."""
        if not self.lock_file.exists():
            if self.mode != "bootstrap":
                self.log(self.lock_file, 1, "T1", "LOCK.json is missing. Run scripts/lock_tests.py --write.")
            return

        try:
            lock_data = json.loads(self.lock_file.read_text(encoding="utf-8"))
        except Exception as e:
            self.log(self.lock_file, 1, "T1", f"Failed to parse LOCK.json: {e}")
            return

        locked_files: Dict[str, str] = lock_data.get("files", {})
        locked_ids: List[str] = lock_data.get("ids", [])

        # Check approval in LOCK_CHANGES.md if hash differs
        approved_changes: Set[str] = set()
        if self.lock_changes_file.exists():
            changes_content = self.lock_changes_file.read_text(encoding="utf-8")
            for match in re.finditer(r"Approved-by:\s*user", changes_content, re.IGNORECASE):
                # find sha in surrounding block
                start = max(0, match.start() - 300)
                end = min(len(changes_content), match.end() + 300)
                window = changes_content[start:end]
                for sha_match in re.findall(r"\b[a-f0-9]{64}\b", window):
                    approved_changes.add(sha_match)

        # Check file hashes
        current_files_map = {f.name: f for f in test_files}
        for fname, expected_hash in locked_files.items():
            if fname not in current_files_map:
                self.log(self.lock_file, 1, "T9", f"Locked test file '{fname}' was deleted or renamed.")
                continue
            cur_path = current_files_map[fname]
            cur_hash = hashlib.sha256(cur_path.read_bytes()).hexdigest()
            if cur_hash != expected_hash:
                if cur_hash not in approved_changes:
                    self.log(
                        cur_path,
                        1,
                        "T1",
                        f"Test lock drift! Hash {cur_hash[:8]} does not match locked {expected_hash[:8]} without approved LOCK_CHANGES entry.",
                    )

        # T9: Check test IDs
        current_ids: Set[str] = set()
        for tf in test_files:
            try:
                tree = ast.parse(tf.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                        current_ids.add(f"{tf.name}::{node.name}")
            except Exception:
                pass

        for lid in locked_ids:
            if lid not in current_ids:
                self.log(self.lock_file, 1, "T9", f"Locked test ID '{lid}' is missing from test suite.")

    def check_t5_red_evidence(self, test_files: List[Path]) -> None:
        """T5: Every test must have an entry in RED_EVIDENCE.md."""
        if self.mode == "bootstrap":
            return
        if not self.red_evidence_file.exists():
            self.log(self.red_evidence_file, 1, "T5", "RED_EVIDENCE.md is missing.")
            return

        evidence_content = self.red_evidence_file.read_text(encoding="utf-8")
        for tf in test_files:
            try:
                tree = ast.parse(tf.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                        if node.name not in evidence_content:
                            self.log(
                                tf,
                                node.lineno,
                                "T5",
                                f"Test '{node.name}' has no registered failure evidence in RED_EVIDENCE.md.",
                            )
            except Exception:
                pass

    def check_t11_colab(self) -> None:
        """T11: Colab heavy test run verified."""
        colab_runs_dir = self.root_dir / "docs" / "rca" / "colab_runs"
        if not colab_runs_dir.exists():
            self.log(colab_runs_dir, 1, "T11", "No colab_runs directory found.")
            return
        run_files = list(colab_runs_dir.glob("*.json"))
        if not run_files:
            self.log(colab_runs_dir, 1, "T11", "No Colab run record JSON found.")
            return
        # check latest run exit_code == 0
        latest = max(run_files, key=lambda f: f.stat().st_mtime)
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            if data.get("exit_code") != 0:
                self.log(latest, 1, "T11", f"Colab run '{latest.name}' failed with exit code {data.get('exit_code')}.")
        except Exception as e:
            self.log(latest, 1, "T11", f"Invalid Colab run record '{latest.name}': {e}")

    def audit_test_file(self, file_path: Path) -> None:
        """Performs AST inspection on an individual test file."""
        code = file_path.read_text(encoding="utf-8")
        lines = code.splitlines()
        try:
            tree = ast.parse(code, filename=str(file_path))
        except SyntaxError as e:
            self.log(file_path, e.lineno or 1, "PARSE", f"Syntax error: {e}")
            return

        # T3: Independent Oracle - file must import sklearn.metrics, numpy, or scipy
        imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

        has_oracle_import = any(
            imp.startswith("sklearn.metrics") or imp.startswith("numpy") or imp.startswith("scipy")
            for imp in imports
        )
        if not has_oracle_import:
            self.log(
                file_path,
                1,
                "T3",
                "Test file lacks independent oracle import (must import sklearn.metrics, numpy, or scipy).",
            )

        # Walk AST for function-level and statement-level rules
        for node in ast.walk(tree):
            # T4: Banned mocks return_value
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "return_value":
                        self.log(file_path, node.lineno, "T4", "Banned 'return_value=' in mock call. Mocks may only observe (wraps) or break (side_effect).")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "return_value":
                        self.log(file_path, node.lineno, "T4", "Banned '.return_value =' mock assignment.")

            # T6: Undeclared loose tolerances in pytest.approx
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name == "approx":
                    is_loose = False
                    for kw in node.keywords:
                        if kw.arg in ("rel", "abs") and isinstance(kw.value, (ast.Constant, ast.UnaryOp)):
                            val: Optional[float] = None
                            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
                                val = float(kw.value.value)
                            elif isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.operand, ast.Constant):
                                if isinstance(kw.value.operand.value, (int, float)):
                                    val = -float(kw.value.operand.value)
                            if val is not None and abs(val) > MAX_UNBOUNDED_APPROX:
                                is_loose = True
                                break
                    if is_loose:
                        line_idx = max(0, node.lineno - 1)
                        source_line = lines[line_idx] if line_idx < len(lines) else ""
                        if "# bound:" not in source_line:
                            self.log(
                                file_path,
                                node.lineno,
                                "T6",
                                f"pytest.approx with loose tolerance (> {MAX_UNBOUNDED_APPROX}) requires '# bound: <formula>' comment.",
                            )

            # T2 & T8: Per test function checks
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                self.audit_test_function(file_path, node, lines)

    def audit_test_function(
        self, file_path: Path, fn_node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str]
    ) -> None:
        # T2: Banned decorators (skip, skipif, xfail)
        for dec in fn_node.decorator_list:
            dec_id = ""
            if isinstance(dec, ast.Name):
                dec_id = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_id = dec.attr
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    dec_id = dec.func.id
                elif isinstance(dec.func, ast.Attribute):
                    dec_id = dec.func.attr
            if dec_id in ("skip", "skipif", "xfail"):
                self.log(file_path, dec.lineno, "T2", f"Banned decorator '@{dec_id}' on verification test.")

        # T2: Early return before assertions
        saw_assert = False
        for stmt in fn_node.body:
            if isinstance(stmt, ast.Assert):
                saw_assert = True
            elif isinstance(stmt, (ast.If, ast.While, ast.For)):
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Return) and not saw_assert:
                        self.log(file_path, sub.lineno, "T2", "Banned early return before assertions.")
            elif isinstance(stmt, ast.Return) and not saw_assert:
                self.log(file_path, stmt.lineno, "T2", "Banned early return before assertions.")

        # T2: try/except suppressing AssertionError
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Try):
                has_assert_inside = any(isinstance(child, ast.Assert) for child in ast.walk(sub))
                if has_assert_inside:
                    for handler in sub.handlers:
                        # catching Exception, BaseException, AssertionError, or bare except
                        catches_all = False
                        if handler.type is None:
                            catches_all = True
                        elif isinstance(handler.type, ast.Name) and handler.type.id in (
                            "AssertionError",
                            "Exception",
                            "BaseException",
                        ):
                            catches_all = True
                        if catches_all:
                            self.log(
                                file_path,
                                handler.lineno,
                                "T2",
                                "Banned try/except enclosing assertions that catches AssertionError or Exception.",
                            )

        # T8: Meaningful assertions
        assert_nodes: List[ast.Assert] = [n for n in ast.walk(fn_node) if isinstance(n, ast.Assert)]
        if not assert_nodes:
            self.log(file_path, fn_node.lineno, "T8", f"Test function '{fn_node.name}' contains zero assert statements.")
            return

        meaningful = False
        for an in assert_nodes:
            test_expr = an.test
            # compare expressions (a == b, a <= b, a in b, etc.)
            if isinstance(test_expr, ast.Compare):
                meaningful = True
                break
            # calls to functions (e.g. np.allclose, is_monotone)
            if isinstance(test_expr, ast.Call):
                meaningful = True
                break
            # boolean operations (a and b) where one is comparison
            if isinstance(test_expr, ast.BoolOp):
                if any(isinstance(v, (ast.Compare, ast.Call)) for v in test_expr.values):
                    meaningful = True
                    break

        if not meaningful:
            self.log(
                file_path,
                assert_nodes[0].lineno,
                "T8",
                f"Test function '{fn_node.name}' has only trivial assertions (e.g. assert True or assert obj is not None without comparison).",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Integrity and Anti-Gaming Audit (Gate G0)")
    parser.add_argument("--root", type=str, default=".", help="Workspace root directory")
    parser.add_argument("--test-dir", type=str, default=None, help="Explicit verification tests directory")
    parser.add_argument("--src-dir", type=str, default=None, help="Explicit source directory")
    parser.add_argument("--mode", choices=["enforce", "bootstrap"], default="enforce")
    parser.add_argument("--require-colab", action="store_true", help="Require verified Colab run record")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    vdir = Path(args.test_dir).resolve() if args.test_dir else None
    sdir = Path(args.src_dir).resolve() if args.src_dir else None

    auditor = TestIntegrityAuditor(
        root_dir=root,
        verification_dir=vdir,
        src_dir=sdir,
        mode=args.mode,
        require_colab=args.require_colab,
    )

    violations = auditor.audit_all()
    if violations:
        print(f"[FAIL] Gate G0 FAILED: {len(violations)} test-integrity violation(s) found:\n")
        for v in violations:
            print(f"  {v}")
        return 1

    print("[PASS] Gate G0 PASSED: Test-Integrity contract verified. Zero gaming/suppression detected.")
    return 0



if __name__ == "__main__":
    sys.exit(main())
