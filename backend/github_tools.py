"""
Wraps GitHub's remote MCP server behind plain sync functions. Different
GitHub MCP tools live in different toolsets, each with its own URL:
  - issues    -> search_issues (V3)
  - repos     -> list_commits, get_file_contents, create_branch, push_files (V4/V6)
  - pull_requests -> create_pull_request (V6)
Read-only calls use /readonly URLs where available; write calls (create_branch,
push_files) need the full (non-readonly) repos endpoint.
"""
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from . import config

import re  # add this to the imports at the top of the file, alongside json

GITHUB_ISSUES_MCP_URL = "https://api.githubcopilot.com/mcp/x/issues/readonly"
GITHUB_REPOS_READONLY_MCP_URL = "https://api.githubcopilot.com/mcp/x/repos/readonly"
GITHUB_REPOS_WRITE_MCP_URL = "https://api.githubcopilot.com/mcp/x/repos"
GITHUB_PR_MCP_URL = "https://api.githubcopilot.com/mcp/x/pull_requests"


def parse_owner_repo(repo_url: str) -> tuple[str, str]:
    cleaned = repo_url.rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    parts = cleaned.split("/")
    return parts[-2], parts[-1]


async def _call_tool(mcp_url: str, tool_name: str, arguments: dict):
    headers = {"Authorization": f"Bearer {config.GITHUB_PAT}"}
    async with streamablehttp_client(mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments)

def _extract_json_blocks(result) -> list[dict]:
    """MCP tool results can return content as either plain text blocks or
    resource blocks (confirmed by testing — get_file_contents uses resource,
    list_commits/list_issues use text). Check both so parsing doesn't
    silently fail depending on which shape a given tool happens to use."""
    parsed = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            resource = getattr(block, "resource", None)
            text = getattr(resource, "text", None) if resource else None
        if not text:
            continue
        try:
            data = json.loads(text)
            parsed.append(data)
        except json.JSONDecodeError:
            continue
    return parsed


# ---------- Issues (V3) ----------

async def _search_issues_async(owner: str, repo: str, query: str, limit: int) -> list[dict]:
    result = await _call_tool(GITHUB_ISSUES_MCP_URL, "search_issues", {
        "query": query, "owner": owner, "repo": repo, "perPage": limit,
        "fields": ["number", "title", "state", "html_url", "body"],
    })
    issues = []
    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            issues.extend(data)
        elif isinstance(data, dict) and "items" in data:
            issues.extend(data["items"])
    return issues[:limit]


def search_issues(repo_url: str, query: str, limit: int = 5) -> list[dict]:
    owner, repo = parse_owner_repo(repo_url)
    try:
        return asyncio.run(_search_issues_async(owner, repo, query, limit))
    except Exception as e:
        raise RuntimeError(f"GitHub issue search failed: {e}") from e


# ---------- Commits (V4) ----------

async def _list_commits_async(owner: str, repo: str, limit: int, since) -> list[dict]:
    params = {"owner": owner, "repo": repo, "perPage": limit}
    if since:
        params["since"] = since
    result = await _call_tool(GITHUB_REPOS_READONLY_MCP_URL, "list_commits", params)
    commits = []
    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            commits.extend(data)
    return commits[:limit]


def list_recent_commits(repo_url: str, limit: int = 5, since=None) -> list[dict]:
    owner, repo = parse_owner_repo(repo_url)
    try:
        return asyncio.run(_list_commits_async(owner, repo, limit, since))
    except Exception as e:
        raise RuntimeError(f"GitHub commit lookup failed: {e}") from e


# ---------- File contents (V6) ----------

async def _get_file_contents_async(owner: str, repo: str, path: str) -> str:
    result = await _call_tool(GITHUB_REPOS_READONLY_MCP_URL, "get_file_contents", {
        "owner": owner, "repo": repo, "path": path,
    })
    for block in result.content:
        resource = getattr(block, "resource", None)
        if resource is not None and getattr(resource, "text", None) is not None:
            return resource.text
    raise RuntimeError(f"No file content found in response for {path}")


def get_file_contents(repo_url: str, path: str) -> str:
    owner, repo = parse_owner_repo(repo_url)
    try:
        return asyncio.run(_get_file_contents_async(owner, repo, path))
    except Exception as e:
        raise RuntimeError(f"Could not fetch file contents for {path}: {e}") from e


# ---------- Write tools (V6) ----------

async def _create_branch_async(owner: str, repo: str, branch: str) -> None:
    result = await _call_tool(GITHUB_REPOS_WRITE_MCP_URL, "create_branch", {
        "owner": owner, "repo": repo, "branch": branch,
    })
    if result.isError:
        raise RuntimeError(f"create_branch failed: {result.content}")


async def _push_files_async(owner: str, repo: str, branch: str, files: list[dict], message: str) -> None:
    result = await _call_tool(GITHUB_REPOS_WRITE_MCP_URL, "push_files", {
        "owner": owner, "repo": repo, "branch": branch, "files": files, "message": message,
    })
    if result.isError:
        raise RuntimeError(f"push_files failed: {result.content}")


async def _create_pull_request_async(owner: str, repo: str, title: str, body: str, head: str, base: str) -> str:
    result = await _call_tool(GITHUB_PR_MCP_URL, "create_pull_request", {
        "owner": owner, "repo": repo, "title": title, "body": body, "head": head, "base": base,
    })
    if result.isError:
        raise RuntimeError(f"create_pull_request failed: {result.content}")

    # Try structured JSON first
    for data in _extract_json_blocks(result):
        if isinstance(data, dict) and "html_url" in data:
            return data["html_url"]

    # Fall back to scanning raw text for a PR URL, in case the tool returned
    # a plain confirmation sentence instead of JSON
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            resource = getattr(block, "resource", None)
            text = getattr(resource, "text", None) if resource else None
        if text:
            match = re.search(r"https://github\.com/[\w.-]+/[\w.-]+/pull/\d+", text)
            if match:
                return match.group(0)

    return "(created, but PR URL could not be parsed from response)"


def create_branch(repo_url: str, branch: str) -> None:
    owner, repo = parse_owner_repo(repo_url)
    asyncio.run(_create_branch_async(owner, repo, branch))


def push_files(repo_url: str, branch: str, files: list[dict], message: str) -> None:
    owner, repo = parse_owner_repo(repo_url)
    asyncio.run(_push_files_async(owner, repo, branch, files, message))


def create_pull_request(repo_url: str, title: str, body: str, head: str, base: str) -> str:
    owner, repo = parse_owner_repo(repo_url)
    return asyncio.run(_create_pull_request_async(owner, repo, title, body, head, base))