"""Machine-checked Quality Gate Scorecard for ml-mcp (Gates G1 - G10).

This script objectively evaluates all 10 Quality Gates.
No human claims or manual overrides are permitted.
Exit code = number of failed gates (0 = 100/100 PASS).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GateResult:
    gate_id: str
    name: str
    passed: bool
    evidence: str
    details: Optional[Dict[str, Any]] = None


class QualityGateRunner:
    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.root = workspace_root or Path(__file__).resolve().parent.parent
        self.src_dir = self.root / "src" / "ml_mcp"
        self.results: List[GateResult] = []

    def run_all(self) -> int:
        print("=" * 70)
        print("EXECUTING MACHINE QUALITY GATE (G1 - G10)")
        print(f"Workspace: {self.root}")
        print("=" * 70)

        checks = [
            self.check_g0,
            self.check_g1,
            self.check_g2,
            self.check_g3,
            self.check_g4,
            self.check_g5,
            self.check_g6,
            self.check_g7,
            self.check_g8,
            self.check_g9,
            self.check_g10,
        ]

        for check in checks:
            try:
                res = check()
                self.results.append(res)
                status = "PASS [OK]" if res.passed else "FAIL [X]"
                print(f"[{res.gate_id}] {res.name:<45} : {status}", flush=True)
                if not res.passed:
                    print(f"     -> {res.evidence}", flush=True)

            except Exception as e:
                self.results.append(
                    GateResult(
                        gate_id=check.__name__.replace("check_", "").upper(),
                        name=check.__name__,
                        passed=False,
                        evidence=f"Gate check crashed: {e}",
                    )
                )

        report_path = self.root / "gate_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        self._print_markdown_table()
        failed_count = sum(1 for r in self.results if not r.passed)
        total_count = len(self.results)
        print(f"\nFinal Verdict: {total_count - failed_count}/{total_count} gates passed. Exit code: {failed_count}", flush=True)
        return failed_count

 
    def check_g0(self) -> GateResult:
        """G0: Test Integrity & Anti-Gaming Gate (AST inspection)."""
        check_script = self.root / "scripts" / "check_test_integrity.py"
        if not check_script.exists():
            return GateResult("G0", "Test-Integrity Anti-Gaming Gate", False, "check_test_integrity.py not found")
        lock_file = self.root / "tests" / "verification" / "LOCK.json"
        cmd = [sys.executable, str(check_script)]
        if not lock_file.exists():
            cmd.extend(["--mode", "bootstrap"])
        res = subprocess.run(cmd, capture_output=True, text=True)
        passed = res.returncode == 0
        evidence = res.stdout.strip() if passed else (res.stderr.strip() or res.stdout.strip())
        return GateResult("G0", "Test-Integrity Anti-Gaming Gate", passed, evidence)

    def check_g1(self) -> GateResult:
        """G1: No silent exception handling (AST check)."""
        check_script = self.root / "scripts" / "check_no_silent_except.py"
        if not check_script.exists():
            return GateResult("G1", "Zero Silent Handlers Gate", False, "check_no_silent_except.py not found")
        cmd = [sys.executable, str(check_script), str(self.src_dir)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        passed = res.returncode == 0
        evidence = res.stdout.strip() if passed else (res.stderr.strip() or res.stdout.strip())
        return GateResult("G1", "Zero Silent Handlers Gate", passed, evidence)

    def check_g2(self) -> GateResult:
        """G2: No hardcoded metrics or guarantee slack."""
        violations = []
        metric_keywords = {"score", "auc", "ece", "brier", "coverage", "risk", "latency", "accuracy", "f1"}

        for py_file in self.src_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except Exception as e:
                violations.append(f"{py_file.name}: Syntax error: {e}")
                continue

            for node in ast.walk(tree):
                # Check for dict key = metric with literal float value
                if isinstance(node, ast.Dict):
                    for k, v in zip(node.keys, node.values):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            if k.value.lower() in metric_keywords and isinstance(v, ast.Constant) and isinstance(v.value, float):
                                violations.append(f"{py_file.name}:{node.lineno}: Hardcoded metric '{k.value}': {v.value}")
                # Check for + 0.05 on guarantee comparisons
                if isinstance(node, ast.Compare):
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.BinOp) and isinstance(comparator.op, ast.Add):
                            if isinstance(comparator.right, ast.Constant) and isinstance(comparator.right.value, float):
                                violations.append(f"{py_file.name}:{node.lineno}: Guarantee comparison has literal slack: + {comparator.right.value}")

        test_file = self.root / "tests" / "verification" / "test_no_fake_metrics.py"
        if test_file.exists():
            res = subprocess.run([sys.executable, "-m", "pytest", str(test_file), "-q"], capture_output=True, text=True, cwd=str(self.root))
            if res.returncode != 0:
                violations.append(f"test_no_fake_metrics.py failed: {res.stdout.strip()[-120:]}")

        passed = len(violations) == 0
        evidence = "No hardcoded metrics or guarantee slack found in src/ml_mcp" if passed else f"Found {len(violations)} violations: {'; '.join(violations[:3])}"
        return GateResult("G2", "No Hardcoded Metrics / Zero Guarantee Slack", passed, evidence, {"violations": violations})

    def check_g3(self) -> GateResult:
        """G3: Every reported metric is out-of-sample (shuffled label tests)."""
        test_file = self.root / "tests" / "verification" / "test_out_of_sample.py"
        if not test_file.exists():
            return GateResult("G3", "Honest Out-Of-Sample Metrics", False, "FAIL: test_out_of_sample.py not implemented yet (scheduled P1)")
        res = subprocess.run([sys.executable, "-m", "pytest", str(test_file), "-q"], capture_output=True, text=True, cwd=str(self.root))
        passed = res.returncode == 0
        evidence = res.stdout.strip() if passed else f"Out-of-sample tests failed: {res.stdout.strip()[-150:]}"
        return GateResult("G3", "Honest Out-Of-Sample Metrics", passed, evidence)

    def check_g4(self) -> GateResult:
        """G4: Fallbacks are visible in warnings list."""
        test_file = self.root / "tests" / "verification" / "test_fallback_warnings.py"
        if not test_file.exists():
            return GateResult("G4", "Visible Fallbacks in Warnings", False, "FAIL: test_fallback_warnings.py not implemented yet (scheduled P1)")
        res = subprocess.run([sys.executable, "-m", "pytest", str(test_file), "-q"], capture_output=True, text=True, cwd=str(self.root))
        passed = res.returncode == 0
        evidence = res.stdout.strip() if passed else f"Fallback warnings tests failed: {res.stdout.strip()[-150:]}"
        return GateResult("G4", "Visible Fallbacks in Warnings", passed, evidence)

    def check_g5(self) -> GateResult:
        """G5: Artifacts are real fitted estimators (round-trip)."""
        test_file = self.root / "tests" / "verification" / "test_artifact_roundtrip.py"
        if not test_file.exists():
            return GateResult("G5", "Fitted Estimators Round-Trip", False, "FAIL: test_artifact_roundtrip.py not implemented yet (scheduled P1)")
        res = subprocess.run([sys.executable, "-m", "pytest", str(test_file), "-q"], capture_output=True, text=True, cwd=str(self.root))
        passed = res.returncode == 0
        evidence = res.stdout.strip() if passed else f"Round-trip tests failed: {res.stdout.strip()[-150:]}"
        return GateResult("G5", "Fitted Estimators Round-Trip", passed, evidence)

    def check_g6(self) -> GateResult:
        """G6: Statistical guarantees are honest (Wilson CI, min-n, multi-seed)."""
        test_file = self.root / "tests" / "verification" / "test_coverage_validity.py"
        if not test_file.exists():
            return GateResult("G6", "Statistical Guarantees Honesty & Validity", False, "FAIL: test_coverage_validity.py not implemented yet (scheduled P6)")
        res = subprocess.run([sys.executable, "-m", "pytest", str(test_file), "-q"], capture_output=True, text=True, cwd=str(self.root))
        passed = res.returncode == 0
        evidence = res.stdout.strip() if passed else f"Validity tests failed: {res.stdout.strip()[-150:]}"
        return GateResult("G6", "Statistical Guarantees Honesty & Validity", passed, evidence)

    def check_g7(self) -> GateResult:
        """G7: Test suite 100% green and post-refactor engine/services coverage >= 85%."""
        cmd = [sys.executable, "-m", "pytest", "-q", "--disable-warnings"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.root))
        passed = res.returncode == 0
        evidence = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr.strip()
        return GateResult("G7", "Test Suite Green & High Coverage", passed, evidence)

    def check_g8(self) -> GateResult:
        """G8: Ruff Linter & Strict Mypy Typing Gate."""
        ruff_res = subprocess.run([sys.executable, "-m", "ruff", "check", str(self.src_dir)], capture_output=True, text=True)
        mypy_res = subprocess.run(
            [sys.executable, "-m", "mypy", "--config-file", str(self.root / "pyproject.toml"), str(self.src_dir)],
            capture_output=True,
            text=True,
            cwd=str(self.root),
        )
        passed = ruff_res.returncode == 0 and mypy_res.returncode == 0
        evidence = f"Ruff: exit {ruff_res.returncode}; Mypy: exit {mypy_res.returncode}"
        return GateResult("G8", "Ruff Linter & Strict Mypy Type Safety", passed, evidence)

    def check_g9(self) -> GateResult:
        """G9: Architecture: no IO in services, no hardcoded random_state=42 literals."""
        violations = []
        services_dir = self.src_dir / "services"

        if services_dir.exists():
            for py_file in services_dir.glob("*.py"):
                if py_file.name == "base.py":
                    continue
                content = py_file.read_text(encoding="utf-8")
                if "pd.read_csv" in content:
                    violations.append(f"{py_file.name}: contains raw pd.read_csv (violates protocol repository IO)")
                if "joblib.load" in content or "joblib.dump" in content:
                    violations.append(f"{py_file.name}: contains raw joblib calls (violates protocol repository IO)")

        # Count literal random_state=42
        r42_count = 0
        for py_file in self.src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            r42_count += content.count("random_state=42") + content.count("RandomState(42)")

        if r42_count > 0:
            violations.append(f"Found {r42_count} occurrences of hardcoded 'random_state=42' (violates central Settings)")

        passed = len(violations) == 0
        evidence = "Clean Architecture enforced: zero IO in services, zero hardcoded seeds" if passed else f"Violations: {'; '.join(violations[:3])}"
        return GateResult("G9", "3-Tier Clean Architecture Invariants", passed, evidence, {"violations": violations})

    def check_g10(self) -> GateResult:
        """G10: Docs match behavior: every guarantee in ROADMAP has [test: ...] tag."""
        import re
        roadmap = self.root / "ROADMAP.md"
        if not roadmap.exists():
            return GateResult("G10", "Test-Backed Documentation Integrity", False, "ROADMAP.md not found")
        content = roadmap.read_text(encoding="utf-8")
        if "[test:" not in content:
            return GateResult("G10", "Test-Backed Documentation Integrity", False, "FAIL: ROADMAP guarantees are not yet tagged with [test: path::name] (scheduled P7)")

        tags = re.findall(r"\[test:\s*([^\]]+)\]", content)
        missing = []
        for tag in tags:
            tag = tag.strip()
            parts = tag.split("::")
            test_file = self.root / parts[0]
            if not test_file.exists():
                missing.append(f"File not found: {parts[0]}")
            elif len(parts) > 1:
                file_text = test_file.read_text(encoding="utf-8")
                if parts[1] not in file_text:
                    missing.append(f"Function not found: {tag}")

        if missing:
            return GateResult("G10", "Test-Backed Documentation Integrity", False, f"Missing test targets: {'; '.join(missing[:3])}")
        return GateResult("G10", "Test-Backed Documentation Integrity", True, f"All {len(tags)} documentation claim tags verified against test suite")

    def _print_markdown_table(self) -> None:
        print("\n" + "=" * 80)
        print("| Gate | Name | Status | Evidence |")
        print("|:---|:---|:---:|:---|")
        for r in self.results:
            status_tag = "PASS" if r.passed else "FAIL"
            clean_evidence = r.evidence.replace("\n", " ")[:60]
            print(f"| {r.gate_id} | {r.name} | {status_tag} | {clean_evidence} |")
        print("=" * 80 + "\n")



if __name__ == "__main__":
    runner = QualityGateRunner()
    sys.exit(runner.run_all())
