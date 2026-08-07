"""Tool registry for agents — smolagents-style.

Each tool is an async callable with a name, description, and JSON-schema-ish
input spec. Tools run in-process by default; `shell_run` and `python_exec`
shell out to a subprocess so they can be timed-out and killed.

Tools are sandboxed to `workspace_root` (default: cwd). File paths must
resolve under it — attempts to escape raise `ToolSecurityError`.

Usage:
    registry = ToolRegistry()
    result = await registry.call("file_read", path="README.md")

Agents declare which tools they have:
    agent.tools = registry.subset(["file_read", "python_exec"])
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    """Raised when a tool fails."""


class ToolSecurityError(ToolError):
    """Raised when a tool call escapes the sandbox."""


@dataclass
class Tool:
    name: str
    description: str
    inputs: dict[str, str]
    func: Callable[..., Awaitable[Any]]

    async def __call__(self, **kwargs: Any) -> Any:
        return await self.func(**kwargs)


@dataclass
class ToolRegistry:
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    timeout_seconds: float = 30.0
    tools: dict[str, Tool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        self._register_builtins()

    def _resolve_safe(self, path: str) -> Path:
        candidate = (self.workspace_root / path).resolve()
        if self.workspace_root not in candidate.parents and candidate != self.workspace_root:
            raise ToolSecurityError(
                f"path {path!r} escapes workspace {self.workspace_root}"
            )
        return candidate

    def _register_builtins(self) -> None:
        async def file_read(path: str) -> str:
            target = self._resolve_safe(path)
            return target.read_text()

        async def file_write(path: str, content: str) -> dict[str, Any]:
            target = self._resolve_safe(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return {"path": str(target), "bytes": len(content)}

        async def shell_run(cmd: str) -> dict[str, Any]:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(self.workspace_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise ToolError(f"shell_run timed out after {self.timeout_seconds}s") from None
            return {
                "exit_code": proc.returncode,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
            }

        async def python_exec(code: str) -> dict[str, Any]:
            # Run as -c in a subprocess so the parent process state is untouched.
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
                cwd=str(self.workspace_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise ToolError(f"python_exec timed out after {self.timeout_seconds}s") from None
            return {
                "exit_code": proc.returncode,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
            }

        async def list_dir(path: str = ".") -> list[str]:
            target = self._resolve_safe(path)
            return sorted(p.name for p in target.iterdir())

        self.register(Tool(
            name="file_read",
            description="Read a UTF-8 text file from the workspace.",
            inputs={"path": "Relative path under workspace_root"},
            func=file_read,
        ))
        self.register(Tool(
            name="file_write",
            description="Write a UTF-8 text file (creates parent dirs).",
            inputs={"path": "Relative path", "content": "File contents"},
            func=file_write,
        ))
        self.register(Tool(
            name="shell_run",
            description="Run a shell command in the workspace with timeout.",
            inputs={"cmd": "Shell command line"},
            func=shell_run,
        ))
        self.register(Tool(
            name="python_exec",
            description="Run a Python snippet in a subprocess with timeout.",
            inputs={"code": "Python source to run via python3 -c"},
            func=python_exec,
        ))
        self.register(Tool(
            name="list_dir",
            description="List directory entries (names only).",
            inputs={"path": "Relative path, default '.'"},
            func=list_dir,
        ))

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def subset(self, names: list[str]) -> ToolRegistry:
        sub = ToolRegistry(workspace_root=self.workspace_root, timeout_seconds=self.timeout_seconds)
        sub.tools = {n: self.tools[n] for n in names if n in self.tools}
        return sub

    async def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self.tools:
            raise ToolError(f"unknown tool: {name}")
        return await self.tools[name](**kwargs)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "inputs": t.inputs}
            for t in self.tools.values()
        ]
