import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from . import config

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/x/repos"


async def main():
    headers = {"Authorization": f"Bearer {config.GITHUB_PAT}"}
    async with streamablehttp_client(GITHUB_MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_file_contents", {
                "owner": "CATSPHER",
                "repo": "test-repo",
                "path": "README.md",  # or any small real file in that repo
            })
            print("isError:", result.isError)
            for block in result.content:
                text = getattr(block, "text", None)
                print(text[:1500] if text else block)


if __name__ == "__main__":
    asyncio.run(main())