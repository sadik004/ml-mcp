import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ml_mcp.colab_bridge import ColabCloudRunner

def run_test(test_cmd: str):
    runner = ColabCloudRunner()
    code = f"!cd /content/ml_mcp_workspace && {test_cmd}"
    res = runner.execute_code(code, session="gpu", timeout=180.0)
    print("OUTPUT:")
    print(res.get("output"))
    if res.get("error"):
        print("ERROR:")
        print(res.get("error"))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pytest tests/verification/test_out_of_sample.py -v"
    run_test(cmd)
