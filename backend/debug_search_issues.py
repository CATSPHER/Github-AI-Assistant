"""
Run: python -m backend.debug_search_issues
"""
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from . import config
from .github_tools import parse_owner_repo

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/x/repos/readonly"


async def main():
    owner, repo = parse_owner_repo("https://github.com/pallets/flask")
    headers = {"Authorization": f"Bearer {config.GITHUB_PAT}"}
    async with streamablehttp_client(GITHUB_MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_commits", {
                "owner": owner,
                "repo": repo,
                "perPage": 3,
            })
            print("isError:", result.isError)
            for block in result.content:
                text = getattr(block, "text", None)
                print(text[:2000] if text else block)


if __name__ == "__main__":
    asyncio.run(main())