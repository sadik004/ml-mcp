"""Unit tests for G0 Test-Integrity and Anti-Gaming Auditor."""
import hashlib
import json
from pathlib import Path
from scripts.check_test_integrity import TestIntegrityAuditor


def _setup_sandbox(tmp_path: Path) -> dict[str, Path]:
    vdir = tmp_path / "tests" / "verification"
    vdir.mkdir(parents=True, exist_ok=True)
    sdir = tmp_path / "src"
    sdir.mkdir(parents=True, exist_ok=True)
    plans_dir = tmp_path / "docs" / "plans" / "remediation_v3"
    plans_dir.mkdir(parents=True, exist_ok=True)

    red_file = plans_dir / "RED_EVIDENCE.md"
    red_file.write_text("# RED Evidence\n| test_sample | commit | L10 | False |\n", encoding="utf-8")

    changes_file = plans_dir / "LOCK_CHANGES.md"
    changes_file.write_text("# Lock changes\n", encoding="utf-8")

    return {
        "root": tmp_path,
        "vdir": vdir,
        "sdir": sdir,
        "plans": plans_dir,
        "red": red_file,
        "changes": changes_file,
    }


def test_flags_skip_marker(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    test_file.write_text(
        "import numpy as np\n"
        "import pytest\n\n"
        "@pytest.mark.skip\n"
        "def test_sample():\n"
        "    assert 1 == 1\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T2" and "skip" in v.message for v in violations)


def test_flags_early_return(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    test_file.write_text(
        "import numpy as np\n\n"
        "def test_sample():\n"
        "    if True:\n"
        "        return\n"
        "    assert 1 == 1\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T2" and "early return" in v.message for v in violations)


def test_flags_try_around_assert(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    test_file.write_text(
        "import numpy as np\n\n"
        "def test_sample():\n"
        "    try:\n"
        "        assert 1 == 2\n"
        "    except AssertionError:\n"
        "        pass\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T2" and "try/except" in v.message for v in violations)


def test_flags_return_value_mock(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    test_file.write_text(
        "import numpy as np\n"
        "from unittest.mock import patch\n\n"
        "def test_sample():\n"
        "    with patch('some.mod', return_value=0.9):\n"
        "        assert 1 == 1\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T4" and "return_value=" in v.message for v in violations)


def test_allows_wraps_spy(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    test_file.write_text(
        "import numpy as np\n"
        "from unittest.mock import patch\n\n"
        "def test_sample():\n"
        "    with patch('some.mod', wraps=list, autospec=True):\n"
        "        assert 1 == 1\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert not any(v.rule == "T4" for v in violations)


def test_flags_loose_approx(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    test_file.write_text(
        "import numpy as np\n"
        "import pytest\n\n"
        "def test_sample():\n"
        "    assert 0.85 == pytest.approx(0.9, rel=0.05)\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T6" and "bound:" in v.message for v in violations)


def test_allows_declared_bound(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    test_file.write_text(
        "import numpy as np\n"
        "import pytest\n\n"
        "def test_sample():\n"
        "    assert 0.85 == pytest.approx(0.9, rel=0.05)  # bound: 1-alpha-3*se\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert not any(v.rule == "T6" for v in violations)


def test_flags_trivial_assert(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    test_file.write_text(
        "import numpy as np\n\n"
        "def test_sample():\n"
        "    res = {'a': 1}\n"
        "    assert res\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T8" and "trivial assertions" in v.message for v in violations)


def test_flags_src_test_detection(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    src_file = box["sdir"] / "service.py"
    src_file.write_text(
        "import os\n\n"
        "def compute():\n"
        "    if 'PYTEST_CURRENT_TEST' in os.environ:\n"
        "        return 1.0\n"
        "    return 0.0\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="bootstrap",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T7" and "PYTEST_CURRENT_TEST" in v.message for v in violations)


def test_flags_hash_drift(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    initial_content = "import numpy as np\n\ndef test_sample():\n    assert 1 == 1\n"
    test_file.write_text(initial_content, encoding="utf-8")
    initial_hash = hashlib.sha256(initial_content.encode("utf-8")).hexdigest()

    lock_file = box["vdir"] / "LOCK.json"
    lock_file.write_text(
        json.dumps({
            "algorithm": "sha256",
            "files": {"test_r_sample.py": initial_hash},
            "ids": ["test_r_sample.py::test_sample"],
        }),
        encoding="utf-8",
    )

    # Mutate test file
    test_file.write_text(
        "import numpy as np\n\ndef test_sample():\n    assert 2 == 2\n",
        encoding="utf-8",
    )

    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        lock_file=lock_file,
        red_evidence_file=box["red"],
        lock_changes_file=box["changes"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T1" and "Test lock drift" in v.message for v in violations)


def test_flags_removed_id(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    content = "import numpy as np\n\ndef test_sample():\n    assert 1 == 1\n"
    test_file.write_text(content, encoding="utf-8")
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()

    lock_file = box["vdir"] / "LOCK.json"
    lock_file.write_text(
        json.dumps({
            "algorithm": "sha256",
            "files": {"test_r_sample.py": h},
            "ids": ["test_r_sample.py::test_sample", "test_r_sample.py::test_deleted"],
        }),
        encoding="utf-8",
    )

    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        lock_file=lock_file,
        red_evidence_file=box["red"],
        lock_changes_file=box["changes"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T9" and "test_deleted" in v.message for v in violations)


def test_flags_missing_red_evidence(tmp_path: Path) -> None:
    box = _setup_sandbox(tmp_path)
    test_file = box["vdir"] / "test_r_sample.py"
    test_file.write_text(
        "import numpy as np\n\ndef test_unregistered():\n    assert 1 == 1\n",
        encoding="utf-8",
    )
    auditor = TestIntegrityAuditor(
        root_dir=box["root"],
        verification_dir=box["vdir"],
        src_dir=box["sdir"],
        red_evidence_file=box["red"],
        mode="enforce",
    )
    violations = auditor.audit_all()
    assert any(v.rule == "T5" and "test_unregistered" in v.message for v in violations)
