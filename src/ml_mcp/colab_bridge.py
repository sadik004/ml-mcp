"""Colab cloud bridge, tunnel URL persistence, and runtime watchdog."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml_mcp.schemas.colab import ColabSessionDTO


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
        return "import os, sys\n# Bootstrap ml_mcp server\nos.environ['ML_MCP_PORT'] = '{port}'\n!pip install -q ml_mcp\nprint(f'ml_mcp initialized on port {port}')\n".format(port=port)


class ColabCloudRunner:
    """Direct headless orchestrator for Google Colab Cloud GPU and TPU runtimes."""

    def __init__(self) -> None:
        self.base_cmd = self._resolve_cli_cmd()

    @property
    def cli_exe(self) -> str:
        """Backwards compatible reference to resolved binary."""
        return self.base_cmd[0]

    @staticmethod
    def _resolve_cli_cmd() -> List[str]:
        """Resolves portable command invocation for colab-cli."""
        colab_in_path = shutil.which("colab")
        if colab_in_path:
            return [colab_in_path]

        scripts_exe = Path(sys.executable).parent / "Scripts" / "colab.exe"
        if scripts_exe.exists():
            return [str(scripts_exe)]

        # Fallback to python module execution
        return [sys.executable, "-m", "colab_cli.cli"]

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns all configured and discovered Colab compute sessions."""
        sessions: List[Dict[str, Any]] = []
        try:
            from colab_cli.state import StateStore
            store = StateStore()
            raw_list = store.list()
            items = list(raw_list.values()) if isinstance(raw_list, dict) else list(raw_list)
            for s in items:
                sessions.append({
                    "name": getattr(s, "name", "session"),
                    "endpoint": getattr(s, "endpoint", "unknown"),
                    "accelerator": getattr(s, "accelerator", "NONE"),
                    "variant": getattr(s, "variant", "DEFAULT"),
                    "machine_shape": getattr(s, "machine_shape", "STANDARD"),
                    "url": getattr(s, "url", ""),
                })
        except Exception:
            pass

        # If store empty, attempt sync
        if not sessions:
            try:
                from colab_cli.common import state
                _, assignments = state.sync_sessions()
                for a in (assignments or []):
                    sessions.append({
                        "name": getattr(a, "name", "remote"),
                        "endpoint": getattr(a, "endpoint", "unknown"),
                        "accelerator": getattr(getattr(a, "accelerator", None), "value", "NONE"),
                        "variant": getattr(getattr(a, "variant", None), "name", "DEFAULT"),
                        "machine_shape": getattr(getattr(a, "machine_shape", None), "name", "STANDARD"),
                        "url": getattr(getattr(a, "runtime_proxy_info", None), "url", ""),
                    })
            except Exception:
                pass

        return sessions

    def _sync_remote_assignment_if_needed(self, session_name: Optional[str] = "gpu") -> Optional[Dict[str, Any]]:
        """Auto-syncs or auto-provisions remote Colab compute session, prioritizing GPU."""
        try:
            from colab_cli.common import state
            from colab_cli.state import SessionState, StateStore
            from colab_cli.commands import session as session_cmd

            store = StateStore()
            raw_list = store.list()
            all_stored = list(raw_list.values()) if isinstance(raw_list, dict) else list(raw_list)

            # If a specific name requested and exists in store
            if session_name:
                existing = store.get(session_name)
                if existing:
                    return {
                        "name": existing.name,
                        "endpoint": existing.endpoint,
                        "accelerator": getattr(existing, "accelerator", "NONE"),
                        "variant": getattr(existing, "variant", "DEFAULT"),
                        "url": getattr(existing, "url", ""),
                    }

            # If default "gpu" requested or none, prioritize stored sessions with GPU
            if (not session_name or session_name == "gpu") and all_stored:
                gpu_stored = [
                    s for s in all_stored
                    if "gpu" in str(getattr(s, "accelerator", "")).lower()
                    or "t4" in str(getattr(s, "accelerator", "")).lower()
                    or "gpu" in str(getattr(s, "variant", "")).lower()
                    or "t4" in str(getattr(s, "name", "")).lower()
                ]
                selected = gpu_stored[0] if gpu_stored else all_stored[0]
                return {
                    "name": selected.name,
                    "endpoint": selected.endpoint,
                    "accelerator": getattr(selected, "accelerator", "NONE"),
                    "variant": getattr(selected, "variant", "DEFAULT"),
                    "url": getattr(selected, "url", ""),
                }

            # Query server assignments
            _, assignments = state.sync_sessions()
            if assignments:
                a = assignments[0]
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=getattr(a.runtime_proxy_info, "token_expires_in_seconds", 3600)
                )
                target_name = session_name or "gpu"
                new_session = SessionState(
                    name=target_name,
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

            # Auto-provision Tesla T4 if no assignments exist
            try:
                target_name = session_name or "gpu"
                session_cmd.new(session=target_name, gpu="T4")
                existing = store.get(target_name)
                if existing:
                    return {
                        "name": existing.name,
                        "endpoint": existing.endpoint,
                        "accelerator": getattr(existing, "accelerator", "T4"),
                        "variant": getattr(existing, "variant", "GPU"),
                        "url": getattr(existing, "url", ""),
                    }
            except Exception:
                pass

        except Exception:
            pass
        return None

    def provision_session(
        self,
        session_name: str = "gpu",
        accelerator: str = "T4",
        high_mem: bool = False,
    ) -> Dict[str, Any]:
        """Provisions a new dedicated compute session with requested hardware."""
        cmd = [*self.base_cmd, "new", "-s", session_name]
        acc_upper = accelerator.strip().upper()
        if acc_upper in ["T4", "L4", "G4", "H100", "A100"]:
            cmd.extend(["--gpu", acc_upper])
        elif acc_upper.startswith("V"):
            cmd.extend(["--tpu", accelerator.strip()])

        if high_mem:
            cmd.append("--high-mem")

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60.0,
            )
            return {
                "success": res.returncode == 0,
                "session_name": session_name,
                "accelerator": accelerator,
                "output": res.stdout,
                "error": res.stderr if res.returncode != 0 else None,
            }
        except Exception as exc:
            return {"success": False, "session_name": session_name, "error": str(exc)}

    def get_status(self, session_name: Optional[str] = "gpu") -> Dict[str, Any]:
        """Probes and returns true empirical Colab hardware specs, VRAM, and RAM."""
        info = self._sync_remote_assignment_if_needed(session_name)
        active_name = info["name"] if info else (session_name or "gpu")
        if not info:
            dto = ColabSessionDTO(
                session_name=active_name,
                status="disconnected",
                hardware="None",
                cuda_available=False,
                is_alive=False,
                warnings=["No active Colab compute session found in state or cloud assignments."],
            )
            return dto.to_compact()

        probe_code = 'import torch, psutil\ncuda = bool(torch.cuda.is_available())\ndev = str(torch.cuda.get_device_name(0)) if cuda else "CPU"\nvram_tot = float(torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda else 0.0\nvram_use = float(torch.cuda.memory_allocated(0) / (1024**3)) if cuda else 0.0\nram_tot = float(psutil.virtual_memory().total / (1024**3))\nram_avail = float(psutil.virtual_memory().available / (1024**3))\nprint(f"PROBE:CUDA={cuda}|DEV={dev}|VRAM_TOT_GB={vram_tot:.2f}|VRAM_USE_GB={vram_use:.2f}|RAM_TOT_GB={ram_tot:.2f}|RAM_AVAIL_GB={ram_avail:.2f}")\n'

        res = self.execute_code(probe_code, session=active_name, timeout=25.0)

        cuda_available = False
        hardware = "CPU"
        vram_total_mb = 0.0
        vram_alloc_mb = 0.0
        ram_total_gb = 0.0
        ram_avail_gb = 0.0
        warnings: List[str] = []

        if res.get("success"):
            out = res.get("output", "")
            match = re.search(
                r"PROBE:CUDA=(True|False)[|]DEV=([^|]+)[|]VRAM_TOT_GB=([\d.]+)[|]VRAM_USE_GB=([\d.]+)[|]RAM_TOT_GB=([\d.]+)[|]RAM_AVAIL_GB=([\d.]+)",
                out,
            )
            if match:
                cuda_available = match.group(1) == "True"
                hardware = match.group(2).strip()
                vram_total_mb = float(match.group(3)) * 1024.0
                vram_alloc_mb = float(match.group(4)) * 1024.0
                ram_total_gb = float(match.group(5))
                ram_avail_gb = float(match.group(6))
            else:
                warnings.append(f"Could not parse full hardware probe output: {out[:120]}")
        else:
            warnings.append(f"Hardware probe failed on Colab runtime: {res.get('error')}")

        dto = ColabSessionDTO(
            session_name=active_name,
            status="connected" if res.get("success") else "error",
            hardware=hardware,
            cuda_available=cuda_available,
            is_alive=res.get("success", False),
            vram_allocated_mb=vram_alloc_mb,
            vram_total_mb=vram_total_mb,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_avail_gb,
            endpoint=info.get("endpoint"),
            tunnel_url=None,
            active_job_id=None,
            warnings=warnings,
        )

        compact = dto.to_compact()
        # Backwards compatible keys for callers expecting legacy dict format
        compact["raw_output"] = res.get("output", "").strip()
        compact["cuda_verified"] = cuda_available
        return compact

    def execute_code(
        self,
        code: str,
        session: Optional[str] = "gpu",
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Executes Python/shell code directly on the remote Colab VM."""
        info = self._sync_remote_assignment_if_needed(session)
        target_session = info["name"] if info else (session or "gpu")
        cmd = [*self.base_cmd, "exec", "-s", target_session, "--timeout", str(timeout)]

        start_time = time.time()
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
            duration = round(time.time() - start_time, 2)
            return {
                "success": proc.returncode == 0,
                "session": target_session,
                "output": proc.stdout,
                "error": proc.stderr if proc.returncode != 0 else None,
                "exit_code": proc.returncode,
                "execution_time_seconds": duration,
            }
        except subprocess.TimeoutExpired:
            duration = round(time.time() - start_time, 2)
            return {
                "success": False,
                "session": target_session,
                "error": f"Execution timed out after {timeout}s",
                "exit_code": -1,
                "execution_time_seconds": duration,
            }
        except Exception as exc:
            duration = round(time.time() - start_time, 2)
            return {
                "success": False,
                "session": target_session,
                "error": str(exc),
                "exit_code": -1,
                "execution_time_seconds": duration,
            }

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        session: Optional[str] = "gpu",
    ) -> Dict[str, Any]:
        """Uploads a local file to the remote Colab filesystem."""
        local_p = Path(local_path).resolve()
        if not local_p.exists():
            return {
                "success": False,
                "error": f"Local source file does not exist: {local_path}",
                "local_path": str(local_path),
                "remote_path": str(remote_path),
            }

        info = self._sync_remote_assignment_if_needed(session)
        target_session = info["name"] if info else (session or "gpu")
        cmd = [*self.base_cmd, "upload", "-s", target_session, str(local_p), str(remote_path)]

        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return {
            "success": res.returncode == 0,
            "session": target_session,
            "local_path": str(local_p),
            "remote_path": str(remote_path),
            "bytes_transferred": local_p.stat().st_size if res.returncode == 0 else 0,
            "output": res.stdout,
            "error": res.stderr if res.returncode != 0 else None,
        }

    def download_file(
        self,
        remote_path: str,
        local_path: str,
        session: Optional[str] = "gpu",
    ) -> Dict[str, Any]:
        """Downloads a file from the remote Colab filesystem to local storage."""
        local_p = Path(local_path).resolve()
        local_p.parent.mkdir(parents=True, exist_ok=True)

        info = self._sync_remote_assignment_if_needed(session)
        target_session = info["name"] if info else (session or "gpu")
        cmd = [*self.base_cmd, "download", "-s", target_session, str(remote_path), str(local_p)]

        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return {
            "success": res.returncode == 0,
            "session": target_session,
            "remote_path": str(remote_path),
            "local_path": str(local_p),
            "bytes_transferred": local_p.stat().st_size if local_p.exists() else 0,
            "output": res.stdout,
            "error": res.stderr if res.returncode != 0 else None,
        }

    def stop_session(self, session: Optional[str] = "gpu") -> Dict[str, Any]:
        """Stops/releases the active Colab compute VM to save compute units."""
        info = self._sync_remote_assignment_if_needed(session)
        target_session = info["name"] if info else (session or "gpu")
        cmd = [*self.base_cmd, "stop", "-s", target_session]

        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return {
            "success": res.returncode == 0,
            "session": target_session,
            "output": res.stdout,
            "error": res.stderr if res.returncode != 0 else None,
        }
