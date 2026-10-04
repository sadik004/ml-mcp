"""Asynchronous Colab Quality Gate Runner.

Syncs current workspace to active Colab Tesla T4 GPU session,
launches scripts/quality_gate.py in background via nohup,
and streams execution log until completion.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_mcp.colab_bridge import ColabCloudRunner
from scripts.sync_to_colab import sync


def run_colab_gate() -> int:
    print(">>> 1. Syncing local workspace to Colab...")
    sync()

    runner = ColabCloudRunner()
    print(">>> 2. Launching quality_gate.py in background on Colab...")
    launch_code = """
import subprocess, os
with open('/content/gate.log', 'w') as f:
    f.write('GATE_STARTED\\n')
cmd = "nohup python3 scripts/quality_gate.py > /content/gate.log 2>&1 &"
os.system(f"cd /content/ml_mcp_workspace && {cmd}")
print("LAUNCHED")
"""
    res = runner.execute_code(launch_code, session="gpu", timeout=20.0)
    print("Launch status:", res.get("output", "").strip())

    print(">>> 3. Polling /content/gate.log until gate run completes...")
    completed = False
    log_offset = 0
    poll_count = 0
    max_polls = 60  # max 5 mins

    while not completed and poll_count < max_polls:
        time.sleep(5)
        poll_count += 1
        read_code = f"""
import os
if os.path.exists('/content/gate.log'):
    with open('/content/gate.log', 'r') as f:
        f.seek({log_offset})
        new_content = f.read()
        new_pos = f.tell()
    print(f"OFFSET:{{new_pos}}")
    print("CONTENT_START")
    print(new_content)
    print("CONTENT_END")
else:
    print("LOG_NOT_FOUND")
"""
        poll_res = runner.execute_code(read_code, session="gpu", timeout=15.0)
        out = poll_res.get("output", "")

        if "CONTENT_START\n" in out:
            lines = out.split("CONTENT_START\n")[1].split("\nCONTENT_END")[0]
            if lines.strip():
                print(lines, end="")
            for line in out.splitlines():
                if line.startswith("OFFSET:"):
                    log_offset = int(line.split(":")[1])

        if "Final Verdict:" in out or "Exit code:" in out:
            completed = True
            break

    # Download report
    print("\n>>> 4. Fetching gate_report.json from Colab...")
    report_code = """
import os
if os.path.exists('/content/ml_mcp_workspace/gate_report.json'):
    with open('/content/ml_mcp_workspace/gate_report.json') as f:
        print("REPORT_START")
        print(f.read())
        print("REPORT_END")
"""
    rep_res = runner.execute_code(report_code, session="gpu", timeout=10.0)
    rep_out = rep_res.get("output", "")
    if "REPORT_START\n" in rep_out:
        json_str = rep_out.split("REPORT_START\n")[1].split("\nREPORT_END")[0].strip()
        local_report = Path(__file__).resolve().parent.parent / "gate_report.json"
        local_report.write_text(json_str, encoding="utf-8")
        print(f"Saved local report to: {local_report}")

    return 0


if __name__ == "__main__":
    sys.exit(run_colab_gate())
