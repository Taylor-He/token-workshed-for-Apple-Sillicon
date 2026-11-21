"""NextCore-owned MCP connections for the in-app SGLang Agent."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from typing import Any

from ..mcp.manager import MCPClientManager
from ..mcp.types import MCPConfig, MCPServerConfig, MCPTransport
from .store import AgentStateStore


class MCPRuntime:
    """Keep MCP clients on one private event loop and expose sync Agent hooks."""

    def __init__(self, store: AgentStateStore):
        self.store = store
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop_main, name="nextcore-mcp", daemon=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._manager: MCPClientManager | None = None
        self._lock = threading.RLock()
        self._thread.start()
        self._ready.wait(timeout=5)

    def _loop_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    def _submit(self, coro, timeout: float = 45):
        loop = self._loop
        if loop is None or not self._ready.is_set():
            raise RuntimeError("NextCore MCP runtime is not available.")
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

    @staticmethod
    def _keychain_service(server: str, env_name: str) -> str:
        return f"com.taylorhe.tokenworkshed.nextcore.mcp.{server}.{env_name}"

    @classmethod
    def _keychain_get(cls, server: str, env_name: str) -> str | None:
        if sys.platform != "darwin":
            return os.environ.get(env_name)
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-s", cls._keychain_service(server, env_name), "-w"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.rstrip("\n") if result.returncode == 0 else os.environ.get(env_name)

    @classmethod
    def _keychain_set(cls, server: str, env_name: str, value: str) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("MCP credentials are stored in macOS Keychain; credential entry is available on macOS only.")
        subprocess.run(
            ["/usr/bin/security", "add-generic-password", "-U", "-a", server, "-s", cls._keychain_service(server, env_name), "-w", value],
            capture_output=True, text=True, timeout=8, check=True,
        )

    @classmethod
    def _keychain_delete(cls, server: str, env_name: str) -> None:
        if sys.platform == "darwin":
            subprocess.run(
                ["/usr/bin/security", "delete-generic-password", "-s", cls._keychain_service(server, env_name)],
                capture_output=True, text=True, timeout=5, check=False,
            )

    def set_secrets(self, server: str, secrets: dict[str, str]) -> list[str]:
        stored: list[str] = []
        for name, value in secrets.items():
            if not value:
                continue
            if not name or not name.replace("_", "").isalnum() or not name[0].isalpha():
                raise ValueError("MCP environment variable names must be valid identifiers.")
            self._keychain_set(server, name, value)
            stored.append(name)
        return stored

    def delete_secrets(self, server: str, names: list[str]) -> None:
        for name in names:
            self._keychain_delete(server, name)

    def _new_manager(self) -> MCPClientManager:
        configs: dict[str, MCPServerConfig] = {}
        for row in self.store.list_mcp_servers():
            if not row["enabled"]:
                continue
            env = {
                name: value
                for name in row["env_names"]
                if (value := self._keychain_get(row["name"], name)) is not None
            }
            configs[row["name"]] = MCPServerConfig(
                name=row["name"],
                transport=MCPTransport(row["transport"]),
                command=row["command"] or None,
                args=row["args"],
                env=env,
                url=row["url"] or None,
                enabled=True,
                timeout=float(row["timeout"]),
            )
        return MCPClientManager(MCPConfig(servers=configs, default_timeout=30))

    async def _ensure_started(self) -> MCPClientManager:
        with self._lock:
            if self._manager is None:
                self._manager = self._new_manager()
        await self._manager.start()
        return self._manager

    async def _reload(self) -> None:
        manager, self._manager = self._manager, None
        if manager is not None:
            await manager.stop()
        await self._ensure_started()

    def reload(self) -> None:
        self._submit(self._reload())

    async def _tools(self) -> list[dict[str, Any]]:
        manager = await self._ensure_started()
        return manager.get_all_tools_openai()

    def tools(self) -> list[dict[str, Any]]:
        try:
            return self._submit(self._tools())
        except Exception:
            return []

    async def _execute(self, name: str, arguments: dict[str, Any]) -> str:
        manager = await self._ensure_started()
        result = await manager.execute_tool(name, arguments)
        if result.is_error:
            return json.dumps({"error": result.error_message or "MCP tool failed."}, ensure_ascii=False)
        if isinstance(result.content, str):
            return result.content[:32_000]
        return json.dumps(result.content, ensure_ascii=False, default=str)[:32_000]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        return str(self._submit(self._execute(name, arguments), timeout=360))

    async def _statuses(self) -> list[dict[str, Any]]:
        manager = await self._ensure_started()
        return [item.to_dict() for item in manager.get_server_status()]

    def statuses(self) -> list[dict[str, Any]]:
        try:
            return self._submit(self._statuses())
        except Exception as exc:
            return [{"state": "error", "error": type(exc).__name__}]

    async def _stop(self) -> None:
        if self._manager:
            await self._manager.stop()
            self._manager = None

    def close(self) -> None:
        if self._loop and self._loop.is_running():
            try:
                self._submit(self._stop(), timeout=10)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=3)
