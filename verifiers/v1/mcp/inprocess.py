"""Task-scoped tool handlers hosted in the verifier worker, without MCP transport."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING

from verifiers.v1.errors import ToolsetError
from verifiers.v1.harnesses.utils.mcp import mcp_content_to_chat_content

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

    from verifiers.v1.mcp.toolset import Toolset
    from verifiers.v1.task import TaskData
    from verifiers.v1.trace import Trace


class InProcessTool:
    def __init__(self, server: MCPServer, toolset: Toolset, name: str, timeout: float):
        self.server, self.toolset, self.name = server, toolset, name
        self.timeout = timeout

    async def call(self, raw_arguments: str, trace: Trace):
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as error:
            return f"error: invalid JSON in tool arguments ({error}); resend the call with valid JSON"
        if not isinstance(arguments, dict):
            return (
                "error: tool arguments must be a JSON object, got "
                f"{type(arguments).__name__}; resend as an object"
            )
        # Match the remote state channel: failed/cancelled calls do not publish
        # partial state; successful calls replace the rollout's state atomically.
        state = trace.state.model_copy(deep=True)
        try:
            async with asyncio.timeout(self.timeout):
                with self.toolset.bind_state(state):
                    try:
                        result = await self.server.call_tool(self.name, arguments)
                    except Exception as error:  # noqa: BLE001 - match MCP's tool-error boundary
                        # Match MCPServer._handle_call_tool's text error result.
                        # Cancellation is not caught or converted to a model result.
                        return str(error)
        except TimeoutError as error:
            raise ToolsetError(f"in-process tool {self.name!r} timed out") from error
        if not hasattr(result, "content"):
            raise ToolsetError("in-process tools cannot request MCP client input")
        current = asyncio.current_task()
        if trace.is_completed or (current is not None and current.cancelling()):
            raise asyncio.CancelledError
        trace.state = state
        return mcp_content_to_chat_content(result.content)


@asynccontextmanager
async def open_inprocess_tools(toolsets: list[Toolset], task: TaskData, timeout: float):
    from mcp.server.mcpserver import MCPServer

    schemas, dispatch = [], {}
    async with AsyncExitStack() as stack:
        for toolset in toolsets:
            cfg = toolset.config
            if cfg.url or cfg.colocated or cfg.runtime.type != "subprocess":
                raise ToolsetError(
                    "inprocess tools require local task scope without url or colocated"
                )
            await stack.enter_async_context(toolset._exit_stack)
            await toolset.setup()
            await toolset.setup_task(task.model_copy(deep=True))
            # Use the exact registration, schemas, argument validation and output
            # conversion used by the MCP backend; never start its HTTP application.
            server = MCPServer(toolset.server_name or type(toolset).__name__)
            toolset.register(server)
            for tool in await server.list_tools():
                full = (
                    f"{toolset.server_name}_{tool.name}"
                    if toolset.server_name
                    else tool.name
                )
                if full in dispatch:
                    raise ToolsetError(f"duplicate in-process tool name {full!r}")
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": full,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
                dispatch[full] = InProcessTool(server, toolset, tool.name, timeout)
        yield schemas, dispatch
