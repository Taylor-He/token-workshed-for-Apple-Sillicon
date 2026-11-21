# SPDX-License-Identifier: Apache-2.0
"""
MCP client for connecting to individual MCP servers.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .types import (
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for connecting to a single MCP server.

    Supports both stdio and SSE transports.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self._session = None
        self._read = None
        self._write = None
        self._tools: List[MCPTool] = []
        self._state = MCPServerState.DISCONNECTED
        self._error: Optional[str] = None
        self._last_connected: Optional[float] = None
        self._lock = asyncio.Lock()
        self._connection_task: asyncio.Task | None = None
        self._connection_ready: asyncio.Event | None = None
        self._disconnect_requested: asyncio.Event | None = None

    @property
    def name(self) -> str:
        """Get server name."""
        return self.config.name

    @property
    def state(self) -> MCPServerState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._state == MCPServerState.CONNECTED

    @property
    def tools(self) -> List[MCPTool]:
        """Get discovered tools."""
        return self._tools

    def get_status(self) -> MCPServerStatus:
        """Get server status."""
        return MCPServerStatus(
            name=self.name,
            state=self._state,
            transport=self.config.transport,
            tools_count=len(self._tools),
            error=self._error,
            last_connected=self._last_connected,
        )

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        Returns:
            True if connection successful, False otherwise
        """
        async with self._lock:
            if self._state == MCPServerState.CONNECTED:
                return True

            if not self.config.enabled:
                logger.info(f"MCP server '{self.name}' is disabled")
                return False

            if self._connection_task is None or self._connection_task.done():
                self._state = MCPServerState.CONNECTING
                self._error = None
                self._connection_ready = asyncio.Event()
                self._disconnect_requested = asyncio.Event()
                self._connection_task = asyncio.create_task(
                    self._connection_lifecycle(),
                    name=f"nextcore-mcp-lifecycle-{self.name[:48]}",
                )
            ready = self._connection_ready
            task = self._connection_task

        if ready is None or task is None:
            return False
        try:
            await asyncio.wait_for(asyncio.shield(ready.wait()), timeout=max(5.0, float(self.config.timeout)))
        except asyncio.TimeoutError:
            logger.error("MCP server '%s' did not connect before its startup timeout", self.name)
            await self._request_lifecycle_stop(task)
            self._state = MCPServerState.ERROR
            self._error = "MCP server startup timed out."
            return False
        return self.is_connected

    async def _connection_lifecycle(self) -> None:
        """Own MCP context entry and exit in one asyncio task.

        AnyIO-backed stdio and SSE context managers require their cancel scopes
        to exit in the task that entered them. Keeping this task alive for the
        connection lifetime avoids cross-task cancel-scope errors on reload or
        application shutdown.
        """
        ready = self._connection_ready
        disconnect_requested = self._disconnect_requested
        try:
            if self.config.transport == MCPTransport.STDIO:
                await self._connect_stdio()
            elif self.config.transport == MCPTransport.SSE:
                await self._connect_sse()
            else:
                raise ValueError(f"Unknown transport: {self.config.transport}")

            await self._initialize_session()
            await self._discover_tools()
            self._state = MCPServerState.CONNECTED
            self._last_connected = time.time()
            logger.info(
                "Connected to MCP server '%s' (%s tools available)",
                self.name,
                len(self._tools),
            )
            if ready is not None:
                ready.set()
            if disconnect_requested is not None:
                await disconnect_requested.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._state = MCPServerState.ERROR
            self._error = str(exc)
            logger.error("Failed to connect to MCP server '%s': %s", self.name, exc)
        finally:
            await self._exit_connection_contexts()
            if self._state != MCPServerState.ERROR:
                self._state = MCPServerState.DISCONNECTED
            self._tools = []
            if ready is not None:
                ready.set()

    async def _request_lifecycle_stop(self, task: asyncio.Task) -> None:
        if self._disconnect_requested is not None:
            self._disconnect_requested.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    async def _exit_connection_contexts(self) -> None:
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
                self._session = None

            if hasattr(self, "_stdio_client") and self._stdio_client:
                await self._stdio_client.__aexit__(None, None, None)
                self._stdio_client = None

            if hasattr(self, "_sse_client") and self._sse_client:
                await self._sse_client.__aexit__(None, None, None)
                self._sse_client = None
        except Exception as exc:
            logger.warning("Error disconnecting from '%s': %s", self.name, exc)

    async def _connect_stdio(self):
        """Connect via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        # Security: Log the command being executed for audit trail
        logger.info(
            f"MCP SECURITY AUDIT: Server '{self.name}' executing command: "
            f"{self.config.command} {' '.join(self.config.args or [])}"
        )

        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env,
        )

        # Create stdio client context
        self._stdio_client = stdio_client(server_params)
        self._read, self._write = await self._stdio_client.__aenter__()

        # Create session
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()

    async def _connect_sse(self):
        """Connect via SSE transport."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        # Create SSE client context
        self._sse_client = sse_client(self.config.url)
        self._read, self._write = await self._sse_client.__aenter__()

        # Create session
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()

    async def _initialize_session(self):
        """Initialize the MCP session."""
        if self._session is None:
            raise RuntimeError("Session not created")

        # Initialize with capabilities
        result = await self._session.initialize()
        logger.debug(
            f"MCP server '{self.name}' initialized: "
            f"protocol={result.protocolVersion}, "
            f"server={result.serverInfo.name if result.serverInfo else 'unknown'}"
        )

    async def _discover_tools(self):
        """Discover available tools from the server."""
        if self._session is None:
            raise RuntimeError("Session not initialized")

        try:
            result = await self._session.list_tools()
            self._tools = []

            for tool in result.tools:
                mcp_tool = MCPTool(
                    server_name=self.name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=(
                        tool.inputSchema if hasattr(tool, "inputSchema") else {}
                    ),
                )
                self._tools.append(mcp_tool)
                logger.debug(f"Discovered tool: {mcp_tool.full_name}")

        except Exception as e:
            logger.warning(f"Failed to discover tools from '{self.name}': {e}")
            self._tools = []

    async def disconnect(self):
        """Disconnect from the MCP server."""
        task = self._connection_task
        if task is None:
            self._state = MCPServerState.DISCONNECTED
            self._tools = []
            return
        if not task.done():
            await self._request_lifecycle_stop(task)
        else:
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                pass
        self._state = MCPServerState.DISCONNECTED
        self._tools = []
        logger.info("Disconnected from MCP server '%s'", self.name)

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> MCPToolResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool (without server prefix)
            arguments: Tool arguments
            timeout: Optional timeout in seconds

        Returns:
            MCPToolResult with the result or error
        """
        if not self.is_connected:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Not connected to server '{self.name}'",
            )

        if self._session is None:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message="Session not initialized",
            )

        try:
            # Call with timeout
            timeout = timeout or self.config.timeout

            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=timeout,
            )

            # Extract content from result
            content = self._extract_content(result)

            return MCPToolResult(
                tool_name=tool_name,
                content=content,
                is_error=result.isError if hasattr(result, "isError") else False,
            )

        except asyncio.TimeoutError:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Tool call timed out after {timeout}s",
            )
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=str(e),
            )

    def _extract_content(self, result) -> Any:
        """Extract content from MCP tool result."""
        if not hasattr(result, "content") or not result.content:
            return None

        # Handle list of content items
        contents = []
        for item in result.content:
            if hasattr(item, "text"):
                contents.append(item.text)
            elif hasattr(item, "data"):
                contents.append(item.data)
            else:
                contents.append(str(item))

        # Return single item or list
        if len(contents) == 1:
            return contents[0]
        return contents

    async def refresh_tools(self):
        """Refresh the list of available tools."""
        if not self.is_connected:
            return

        await self._discover_tools()
