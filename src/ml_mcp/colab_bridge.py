"""Colab cloud bridge, tunnel URL persistence, and runtime watchdog."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


class ColabBridge:
    """Manages Drive persistence, tunnel sync, and bootstrap scripting for Google Colab."""

    def __init__(self, drive_root: str = "/content/drive/MyDrive/ml_mcp") -> None:
        self.drive_root = Path(drive_root).resolve()
        self.checkpoints_dir = self.drive_root / "checkpoints"
        self.artifacts_dir = self.drive_root / "artifacts"
        self.tunnel_file = self.drive_root / "tunnel_url.txt"
        self.init_directories()

    def init_directories(self) -> None:
        """Initializes checkpoint and artifact directories."""
        try:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @property
    def is_initialized(self) -> bool:
        """Checks if storage directories exist."""
        return self.checkpoints_dir.exists() and self.artifacts_dir.exists()

    def persist_tunnel_url(self, url: str) -> None:
        """Atomically saves the active Cloudflare/ngrok/localtunnel SSE URL."""
        self.tunnel_file.write_text(url.strip(), encoding="utf-8")

    def get_persisted_tunnel_url(self) -> Optional[str]:
        """Reads the currently persisted tunnel URL if available."""
        if self.tunnel_file.exists():
            return self.tunnel_file.read_text(encoding="utf-8").strip()
        return None

    def generate_colab_bootstrap_cell(self, port: int = 8000) -> str:
        """Generates bootstrap Python code cell for Colab runtime setup."""
        return (
            f"import os, sys\n"
            f"# Bootstrap ml_mcp server\n"
            f"os.environ['ML_MCP_PORT'] = '{port}'\n"
            f"!pip install -q ml_mcp\n"
            f"print(f'ml_mcp initialized on port {port}')\n"
        )


class ColabCloudRunner:
    """Direct headless orchestrator for Google Colab Cloud GPU runtimes."""

    def __init__(self) -> None:
        self.cli_exe = self._resolve_cli()

    @staticmethod
    def _resolve_cli() -> str:
        """Resolves path to colab.exe or fallback."""
        colab_in_path = shutil.which("colab")
        if colab_in_path:
            return colab_in_path

        scripts_dir = Path(sys.executable).parent / "Scripts" / "colab.exe"
        if scripts_dir.exists():
            return str(scripts_dir)

        return "colab"

    def _sync_remote_assignment_if_needed(self, session_name: str = "gpu") -> Optional[Dict[str, Any]]:
        """Auto-syncs or auto-provisions remote Colab GPU session."""
        try:
            from colab_cli.common import state
            from colab_cli.state import SessionState, StateStore
            from colab_cli.commands import session as session_cmd

            store = StateStore()
            existing = store.get(session_name)
            if existing:
                return {
                    "name": existing.name,
                    "endpoint": existing.endpoint,
                    "accelerator": existing.accelerator,
                    "variant": existing.variant,
                    "url": existing.url,
                }

            # If not in store, query server assignments
            _, assignments = state.sync_sessions()
            if assignments:
                a = assignments[0]
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=getattr(a.runtime_proxy_info, "token_expires_in_seconds", 3600)
                )
                new_session = SessionState(
                    name=session_name,
                    endpoint=a.endpoint,
                    token=a.runtime_proxy_info.token,
                    url=a.runtime_proxy_info.url,
                    variant=a.variant.name,
                    accelerator=a.accelerator.value,
                    machine_shape=a.machine_shape.name,
                    token_expires_at=expires_at,
                )
                store.add(new_session)
                return {
                    "name": new_session.name,
                    "endpoint": new_session.endpoint,
                    "accelerator": new_session.accelerator,
                    "variant": new_session.variant,
                    "url": new_session.url,
                }

            # If no assignments exist at all on Colab, auto-provision Tesla T4!
            try:
                session_cmd.new(session=session_name, gpu="T4")
                existing = store.get(session_name)
                if existing:
                    return {
                        "name": existing.name,
                        "endpoint": existing.endpoint,
                        "accelerator": existing.accelerator,
                        "variant": existing.variant,
                        "url": existing.url,
                    }
            except Exception:
                pass

        except Exception:
            pass
        return None

    def get_status(self, session_name: str = "gpu") -> Dict[str, Any]:
        """Returns active Colab session status, hardware specs, and connection health."""
        info = self._sync_remote_assignment_if_needed(session_name)
        if not info:
            return {"status": "disconnected", "session": session_name, "hardware": None}

        res = self.execute_code(
            'import torch; print("CUDA=" + str(torch.cuda.is_available()) + "|DEV=" + str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"))',
            session=session_name,
            timeout=15.0,
        )

        return {
            "status": "connected" if res.get("success") else "error",
            "session": session_name,
            "hardware": info.get("accelerator", "T4"),
            "endpoint": info.get("endpoint"),
            "cuda_verified": res.get("success", False),
            "raw_output": res.get("output", "").strip(),
        }

    def execute_code(
        self,
        code: str,
        session: str = "gpu",
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Executes code on the remote Colab GPU VM using colab exec."""
        self._sync_remote_assignment_if_needed(session)
        cmd = [self.cli_exe, "exec", "-s", session, "--timeout", str(timeout)]

        try:
            proc = subprocess.run(
                cmd,
                input=code,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout + 15,
            )
            return {
                "success": proc.returncode == 0,
                "output": proc.stdout,
                "error": proc.stderr if proc.returncode != 0 else None,
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Execution timed out after {timeout}s", "exit_code": -1}
        except Exception as exc:
            return {"success": False, "error": str(exc), "exit_code": -1}

    def upload_file(self, local_path: str, remote_path: str, session: str = "gpu") -> Dict[str, Any]:
        """Uploads a local file to the remote Colab filesystem."""
        cmd = [self.cli_exe, "upload", "-s", session, str(local_path), str(remote_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return {
            "success": res.returncode == 0,
            "local_path": str(local_path),
            "remote_path": str(remote_path),
            "output": res.stdout,
            "error": res.stderr if res.returncode != 0 else None,
        }

    def download_file(self, remote_path: str, local_path: str, session: str = "gpu") -> Dict[str, Any]:
        """Downloads a file from the remote Colab filesystem to local storage."""
        cmd = [self.cli_exe, "download", "-s", session, str(remote_path), str(local_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return {
            "success": res.returncode == 0,
            "remote_path": str(remote_path),
            "local_path": str(local_path),
            "output": res.stdout,
            "error": res.stderr if res.returncode != 0 else None,
        }

    def stop_session(self, session: str = "gpu") -> Dict[str, Any]:
        """Stops/releases the active Colab VM runtime to save compute credits."""
        cmd = [self.cli_exe, "stop", "-s", session]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return {"success": res.returncode == 0, "output": res.stdout}
