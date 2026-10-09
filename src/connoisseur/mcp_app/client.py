"""Lab 11: MCP client - connection, discovery, roots, and sampling.

Run:  python -m connoisseur.mcp_app.client
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ListRootsResult,
    Root,
    TextContent,
)

from connoisseur import config

SERVER_SCRIPT = str(Path(__file__).parent / "server.py")

# Launch the server with the *same* interpreter (and therefore the same venv) as this client.
server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])


# ------------------------------------------------------------------ roots
def list_roots() -> list[Root]:
    """Limit the server's file access to this project directory."""
    return [Root(uri=config.PROJECT_ROOT.as_uri(), name=config.PROJECT_ROOT.name)]


async def list_roots_callback(context: Any) -> ListRootsResult:
    return ListRootsResult(roots=list_roots())


# --------------------------------------------------------------- sampling
def get_sampling_client():
    """DeepSeek via its Anthropic-compatible endpoint (keeps the Anthropic SDK code shape)."""
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=config.get_api_key(), base_url=config.DEEPSEEK_ANTHROPIC_BASE_URL)


async def handle_sampling(context: Any, params: CreateMessageRequestParams) -> CreateMessageResult:
    """Run an LLM call on behalf of the server and return the result."""
    prompt = params.messages[0].content.text
    print("\n[Sampling] Server requested LLM task:")
    print(f"  Prompt preview: {prompt[:150]}...")

    response = await get_sampling_client().messages.create(
        model=config.TEXT_MODEL,
        max_tokens=params.maxTokens or 200,
        messages=[{"role": "user", "content": prompt}],
    )
    # skip any "thinking" blocks; keep only text blocks
    response_text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    print(f"  LLM Response: {response_text[:100]}...")

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model=config.TEXT_MODEL,
    )


# ----------------------------------------------------------------- helpers
async def call_tool(tool_name: str, arguments: dict) -> dict:
    """Connect to the server, call a tool, and return the parsed JSON result."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots_callback,
        ) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            text = result.content[0].text
            if getattr(result, "isError", False):
                raise RuntimeError(f"Tool '{tool_name}' failed: {text}")
            return json.loads(text)


async def verify_connection() -> None:
    """Connect to the server and verify all expected tools and resources exist."""
    print("=" * 60)
    print("MCP Connection Verification")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots_callback,
        ) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"\nDiscovered {len(tool_names)} tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {(tool.description or '')[:80]}...")

            for expected in ("get_restaurant_info", "recommend_by_vibe", "get_review"):
                assert expected in tool_names, f"FAIL: {expected} not found!"
            print("\nAll required tools verified!")

            resources_result = await session.list_resources()
            print(f"\nDiscovered {len(resources_result.resources)} resources:")
            for resource in resources_result.resources:
                print(f"  - {resource.uri}: {resource.name}")

            roots = list_roots()
            print(f"\nConfigured {len(roots)} roots:")
            for root in roots:
                print(f"  - {root.name}: {root.uri}")


if __name__ == "__main__":
    asyncio.run(verify_connection())
