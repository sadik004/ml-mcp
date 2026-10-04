"""Computes SHA-256 lock and node IDs for tests/verification/ suite."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List


def lock_verification_suite(root_dir: Path, write: bool = False) -> int:
    verification_dir = root_dir / "tests" / "verification"
    red_evidence = root_dir / "docs" / "plans" / "remediation_v3" / "RED_EVIDENCE.md"
    lock_file = verification_dir / "LOCK.json"

    test_files = sorted(list(verification_dir.glob("test_r_*.py")) + list(verification_dir.glob("**/test_r_*.py")))
    test_files = sorted(list({f.resolve() for f in test_files}))

    if not test_files:
        print("[FAIL] No 'test_r_*.py' files found to lock.")
        return 1

    evidence_text = red_evidence.read_text(encoding="utf-8") if red_evidence.exists() else ""

    files_map: Dict[str, str] = {}
    collected_ids: List[str] = []
    missing_evidence: List[str] = []

    for tf in test_files:
        sha = hashlib.sha256(tf.read_bytes()).hexdigest()
        files_map[tf.name] = sha

        tree = ast.parse(tf.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                test_id = f"{tf.name}::{node.name}"
                collected_ids.append(test_id)
                if node.name not in evidence_text:
                    missing_evidence.append(test_id)

    collected_ids.sort()

    if missing_evidence:
        print(f"[FAIL] Cannot lock: {len(missing_evidence)} test(s) missing from RED_EVIDENCE.md:")
        for mid in missing_evidence:
            print(f"  - {mid}")
        return 1

    lock_payload = {
        "algorithm": "sha256",
        "files": files_map,
        "ids": collected_ids,
    }

    if write:
        lock_file.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")
        print(f"[LOCKED] Successfully locked {len(files_map)} files ({len(collected_ids)} tests) into {lock_file.name}")
    else:
        print(json.dumps(lock_payload, indent=2))


    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock verification tests SHA-256 and node IDs")
    parser.add_argument("--root", type=str, default=".", help="Workspace root")
    parser.add_argument("--write", action="store_true", help="Write LOCK.json")
    args = parser.parse_args()

    return lock_verification_suite(Path(args.root).resolve(), write=args.write)


if __name__ == "__main__":
    sys.exit(main())
