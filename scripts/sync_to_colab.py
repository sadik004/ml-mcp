import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ml_mcp.colab_bridge import ColabCloudRunner

def sync():
    zip_path = Path("workspace_sync.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for base in ["src", "tests", "docs", "scripts"]:
            if not os.path.exists(base):
                continue
            for root, _, files in os.walk(base):
                if "__pycache__" in root or ".pytest_cache" in root:
                    continue
                for f in files:
                    if f.endswith((".pyc", ".pyo")):
                        continue
                    p = os.path.join(root, f)
                    z.write(p, p)
        if os.path.exists("pyproject.toml"):
            z.write("pyproject.toml", "pyproject.toml")
        if os.path.exists("ROADMAP.md"):
            z.write("ROADMAP.md", "ROADMAP.md")
        if os.path.exists("requirements.txt"):
            z.write("requirements.txt", "requirements.txt")

    print(f"Zip created: {zip_path.stat().st_size} bytes")
    runner = ColabCloudRunner()
    upload_res = runner.upload_file(str(zip_path), "/content/workspace_sync.zip", session="gpu")
    print("Upload result:", upload_res)

    extract_code = """
import zipfile, os
with zipfile.ZipFile("/content/workspace_sync.zip", "r") as z:
    z.extractall("/content/ml_mcp_workspace")
if os.path.exists("/content/workspace_sync.zip"):
    os.remove("/content/workspace_sync.zip")
print("SYNC_SUCCESS")
"""
    exec_res = runner.execute_code(extract_code, session="gpu")
    print("Remote extract result:", exec_res.get("output"))

    if zip_path.exists():
        zip_path.unlink()

if __name__ == "__main__":
    sync()
