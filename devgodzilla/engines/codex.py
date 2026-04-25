"""
DevGodzilla Codex Engine

OpenAI Codex CLI engine adapter.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

from devgodzilla.logging import get_logger
from devgodzilla.engines.interface import (
    EngineKind,
    EngineMetadata,
    EngineRequest,
    EngineResult,
    SandboxMode,
)
from devgodzilla.engines.cli_adapter import CLIEngine
from devgodzilla.engines.registry import register_engine

logger = get_logger(__name__)


class CodexEngine(CLIEngine):
    """
    Engine adapter for the OpenAI Codex CLI.
    
    Uses `codex exec` command with appropriate model and sandbox settings.
    Supports planning, execution, and QA modes.
    
    Example:
        engine = CodexEngine(default_model="gpt-5.4")
        result = engine.execute(request)
    """

    def __init__(
        self,
        *,
        default_timeout: int = 180,
        default_model: Optional[str] = None,
    ) -> None:
        super().__init__(
            default_timeout=default_timeout,
            default_model=default_model or os.environ.get("DEVGODZILLA_CODEX_MODEL", "gpt-5.4"),
        )

    @property
    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            id="codex",
            display_name="OpenAI Codex CLI",
            kind=EngineKind.CLI,
            default_model=self._default_model,
            description="OpenAI's Codex CLI for code generation",
            capabilities=["plan", "execute", "qa", "multi-file"],
        )

    def _get_command_name(self) -> str:
        return "codex"

    def _sandbox_to_codex(self, sandbox: SandboxMode) -> str:
        """Convert SandboxMode to Codex sandbox string."""
        mapping = {
            SandboxMode.FULL_ACCESS: "danger-full-access",
            SandboxMode.WORKSPACE_WRITE: "workspace-write",
            SandboxMode.READ_ONLY: "read-only",
        }
        return mapping.get(sandbox, "workspace-write")

    def _build_command(
        self,
        req: EngineRequest,
        sandbox: SandboxMode,
    ) -> List[str]:
        """Build codex exec command."""
        model = self._get_model(req)
        if not model:
            raise ValueError("Codex requires a model")
        
        cwd = Path(req.working_dir)
        codex_sandbox = self._sandbox_to_codex(sandbox)
        
        cmd = [
            "codex",
            "exec",
            "-m", model,
            "--cd", str(cwd),
            "--sandbox", codex_sandbox,
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
        ]
        
        # Add optional parameters from extra
        extra = req.extra or {}
        
        if extra.get("output_schema"):
            cmd.extend(["--output-schema", str(extra["output_schema"])])
        
        if extra.get("output_last_message"):
            cmd.extend(["--output-last-message", str(extra["output_last_message"])])
        
        # Read from stdin
        cmd.append("-")
        
        return cmd

    def check_availability(self) -> bool:
        """
        Check if Codex CLI can run in this environment.

        Codex supports either API-key auth or the CLI's ChatGPT login stored in
        ``$HOME/.codex/auth.json``. Set ``DEVGODZILLA_ASSUME_AGENT_AUTH=true`` to
        bypass the auth check for tests or externally managed credentials.
        """
        if not super().check_availability():
            return False

        if os.environ.get("DEVGODZILLA_ASSUME_AGENT_AUTH", "").lower() in ("1", "true", "yes", "on"):
            return True

        if os.environ.get("OPENAI_API_KEY"):
            return True

        cmd_path = self._resolve_command_path()
        try:
            result = subprocess.run(
                [cmd_path, "login", "status"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("codex_login_status_check_failed", extra={"error": str(exc)})
            return False

        status_text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        return result.returncode == 0 and "not logged in" not in status_text

    def _resolve_command_path(self) -> str:
        """Resolve codex from PATH or the same extra dirs used by CLIEngine."""
        cmd_name = self._get_command_name()
        path = shutil.which(cmd_name)
        if path:
            return path

        for directory in self._SYSTEM_CLI_PATHS + self._extra_cli_dirs():
            candidate = os.path.join(directory, cmd_name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return cmd_name


def register_codex_engine(*, default: bool = True) -> CodexEngine:
    """
    Register CodexEngine in the global registry.
    
    Returns the registered engine instance.
    """
    engine = CodexEngine()
    register_engine(engine, default=default)
    return engine
