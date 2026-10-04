"""AST script to check that no except block silently passes without logging, warning, or re-raising."""
import ast
import sys
from pathlib import Path


def is_silent_body(body: list[ast.stmt]) -> bool:
    """Check if the body only contains pass or ellipsis."""
    if len(body) == 1:
        stmt = body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
            return True
    return False


def check_file(file_path: Path) -> list[str]:
    violations = []
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except Exception as e:
        return [f"Could not parse {file_path}: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                # Check for silent pass
                if is_silent_body(handler.body):
                    violations.append(
                        f"{file_path}:{handler.lineno}: Silent exception handler with only `pass`"
                    )
    return violations


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "src" / "ml_mcp"
    all_violations = []
    for py_file in root.rglob("*.py"):
        violations = check_file(py_file)
        all_violations.extend(violations)

    if all_violations:
        print(f"FAILED: Found {len(all_violations)} silent exception handlers:")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASSED: 0 silent exception handlers found in src/ml_mcp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
