"""
Run: python -m backend.debug_get_file
"""
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from . import config
from .github_tools import GITHUB_PR_MCP_URL, parse_owner_repo


async def main():
    owner, repo = parse_owner_repo("https://github.com/CATSPHER/test-repo")
    headers = {"Authorization": f"Bearer {config.GITHUB_PAT}"}
    async with streamablehttp_client(GITHUB_PR_MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_pull_requests", {
                "owner": owner, "repo": repo, "state": "open",
            })
            for block in result.content:
                text = getattr(block, "text", None)
                print(text[:3000] if text else block)


if __name__ == "__main__":
    asyncio.run(main())