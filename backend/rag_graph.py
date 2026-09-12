"""
V5: adds fix-suggestion mode with a critique/retry loop — the first cycle in
the graph, not just a branch. When a question asks for a fix, generate_fix
proposes one, critique_node checks whether it only references files that
were actually in the retrieved context, and loops back to regenerate
(up to MAX_FIX_RETRIES times) if it invented something that isn't there.

Graph shape (fix path):
  route -> ... -> retrieve_code -> generate_fix -> critique
                                        ^                |
                                        └── retry ────────┘ (if ungrounded, under retry limit)
                                                            -> finalize_fix -> END

Graph shape (normal path, unchanged from V4):
  route -> ... -> retrieve_code -> generate -> END
"""
import re
from typing import TypedDict, Optional

from groq import Groq
from langgraph.graph import StateGraph, START, END

from . import config
from .vectorstore import retrieve
from .github_tools import search_issues, list_recent_commits

ISSUE_KEYWORDS = [
    "issue", "issues", "bug", "bugs", "report", "reported", "broken",
    "error", "crash", "complain", "known problem", "feature request",
]
COMMIT_KEYWORDS = [
    "commit", "commits", "recent", "recently", "changed", "changes",
    "history", "who wrote", "who added", "who changed", "last updated",
    "latest change", "active", "activity",
]
FIX_KEYWORDS = [
    "fix", "suggest a fix", "propose a fix", "resolve", "patch",
    "solve this bug", "how would you fix", "write a fix",
]
MAX_FIX_RETRIES = 2


class GraphState(TypedDict):
    repo_url: str
    question: str
    file_path: Optional[str]
    need_issues: bool
    need_commits: bool
    need_fix: bool
    issues: list[dict]
    commits: list[dict]
    contexts: list[dict]
    proposed_fix: str
    fix_retry_count: int
    fix_grounded: bool
    answer: str


_groq_client = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def route_node(state: GraphState) -> GraphState:
    q = state["question"].lower()
    return {
        **state,
        "need_issues": any(k in q for k in ISSUE_KEYWORDS),
        "need_commits": any(k in q for k in COMMIT_KEYWORDS),
        "need_fix": any(k in q for k in FIX_KEYWORDS),
    }


def after_route(state: GraphState) -> str:
    if state["need_issues"]:
        return "fetch_issues"
    if state["need_commits"]:
        return "fetch_commits"
    return "retrieve_code"


def after_issues(state: GraphState) -> str:
    return "fetch_commits" if state["need_commits"] else "retrieve_code"


def after_retrieve(state: GraphState) -> str:
    return "generate_fix" if state["need_fix"] else "generate"


def fetch_issues_node(state: GraphState) -> GraphState:
    try:
        issues = search_issues(state["repo_url"], state["question"], limit=5)
    except RuntimeError:
        issues = []
    return {**state, "issues": issues}


def fetch_commits_node(state: GraphState) -> GraphState:
    try:
        commits = list_recent_commits(state["repo_url"], limit=8)
    except RuntimeError:
        commits = []
    return {**state, "commits": commits}


def retrieve_node(state: GraphState) -> GraphState:
    hits = retrieve(state["repo_url"], state["question"], file_path=state.get("file_path"))
    return {**state, "contexts": hits}


def _build_context_sections(state: GraphState) -> list[str]:
    sections = []
    if state["contexts"]:
        code_block = "\n\n---\n\n".join(
            f"[{h['source_path']}" + (f" :: {h['symbol']}" if h["symbol"] else "") + f"]\n{h['text']}"
            for h in state["contexts"]
        )
        sections.append(f"CODE CONTEXT:\n{code_block}")
    if state["issues"]:
        issues_block = "\n\n---\n\n".join(
            f"[Issue #{i['number']}: {i['title']} ({i['state']})]\n{(i.get('body') or '')[:500]}\n{i.get('html_url', '')}"
            for i in state["issues"]
        )
        sections.append(f"RELATED GITHUB ISSUES:\n{issues_block}")
    if state["commits"]:
        commits_block = "\n\n---\n\n".join(
            f"[Commit {c['sha'][:7]} by {c['commit']['author']['name']} on {c['commit']['author']['date'][:10]}]\n"
            f"{c['commit']['message']}\n{c.get('html_url', '')}"
            for c in state["commits"]
        )
        sections.append(f"RECENT COMMITS:\n{commits_block}")
    return sections


def generate_node(state: GraphState) -> GraphState:
    sections = _build_context_sections(state)
    if not sections:
        return {**state, "answer": "No relevant code, issues, or commits were found for that question."}

    prompt = (
        "You are a senior engineer helping a teammate understand a project's "
        "code, reported problems, and recent activity. Answer ONLY using the "
        "context below. If the context doesn't contain the answer, say so "
        "instead of guessing. Cite file paths / issue numbers / commit SHAs "
        "in square brackets.\n\n"
        + "\n\n".join(sections)
        + f"\n\nQUESTION: {state['question']}\n\nANSWER:"
    )
    client = get_groq_client()
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=900, temperature=0.2,
    )
    return {**state, "answer": completion.choices[0].message.content}


def generate_fix_node(state: GraphState) -> GraphState:
    sections = _build_context_sections(state)
    if not state["contexts"]:
        return {**state, "proposed_fix": "No code context was found to base a fix on."}

    known_paths = ", ".join(sorted({h["source_path"] for h in state["contexts"]}))
    retry_note = ""
    if state["fix_retry_count"] > 0:
        retry_note = (
            "\n\nIMPORTANT: your previous attempt referenced a file that is NOT "
            f"in the code context below. Only reference these exact files: {known_paths}. "
            "If you cannot propose a grounded fix using only these files, say so explicitly "
            "instead of inventing a file path."
        )

    prompt = (
        "You are a senior engineer proposing a code fix. Base your fix ONLY on "
        "the context below — do not invent files, functions, or line numbers "
        "that aren't shown. Present the fix as a unified diff (--- a/path / "
        "+++ b/path / @@ hunk) preceded by a short explanation of the bug and "
        "the fix.\n\n"
        + "\n\n".join(sections)
        + f"\n\nREQUEST: {state['question']}"
        + retry_note
        + "\n\nFIX:"
    )
    client = get_groq_client()
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=900, temperature=0.2,
    )
    return {**state, "proposed_fix": completion.choices[0].message.content}


def critique_node(state: GraphState) -> GraphState:
    """Heuristic grounding check: does the proposed fix only mention paths we
    actually retrieved? No extra LLM call — a cheap regex check keeps this
    loop free to run."""
    # Normalize to forward slashes — diffs always use '/' (git convention)
    # regardless of OS, but retrieved source_path may use '\' on Windows.
    known_paths = {h["source_path"].replace("\\", "/") for h in state["contexts"]}
    mentioned = set(re.findall(r"[-+]{3} [ab]/([^\s]+)", state["proposed_fix"]))
    mentioned |= set(re.findall(r"\[([^\]\s]+\.\w+)", state["proposed_fix"]))
    mentioned = {m.replace("\\", "/") for m in mentioned}

    if not mentioned:
        return {**state, "fix_grounded": True}

    grounded = all(any(m in kp or kp in m for kp in known_paths) for m in mentioned)
    return {**state, "fix_grounded": grounded}


def after_critique(state: GraphState) -> str:
    if state["fix_grounded"]:
        return "finalize_fix"
    if state["fix_retry_count"] < MAX_FIX_RETRIES:
        return "retry"
    return "finalize_fix"  # give up retrying, ship it with a caveat


def finalize_fix_node(state: GraphState) -> GraphState:
    answer = state["proposed_fix"]
    if not state["fix_grounded"]:
        answer += (
            "\n\n⚠️ Note: this fix could not be fully verified against the "
            "retrieved code context after multiple attempts — double-check "
            "file paths before applying it."
        )
    return {**state, "answer": answer}


def increment_retry_node(state: GraphState) -> GraphState:
    return {**state, "fix_retry_count": state["fix_retry_count"] + 1}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("route", route_node)
    graph.add_node("fetch_issues", fetch_issues_node)
    graph.add_node("fetch_commits", fetch_commits_node)
    graph.add_node("retrieve_code", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("generate_fix", generate_fix_node)
    graph.add_node("critique", critique_node)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("finalize_fix", finalize_fix_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, {
        "fetch_issues": "fetch_issues", "fetch_commits": "fetch_commits", "retrieve_code": "retrieve_code",
    })
    graph.add_conditional_edges("fetch_issues", after_issues, {
        "fetch_commits": "fetch_commits", "retrieve_code": "retrieve_code",
    })
    graph.add_edge("fetch_commits", "retrieve_code")
    graph.add_conditional_edges("retrieve_code", after_retrieve, {
        "generate_fix": "generate_fix", "generate": "generate",
    })
    graph.add_edge("generate", END)

    graph.add_edge("generate_fix", "critique")
    graph.add_conditional_edges("critique", after_critique, {
        "retry": "increment_retry", "finalize_fix": "finalize_fix",
    })
    graph.add_edge("increment_retry", "generate_fix")  # <- the actual loop
    graph.add_edge("finalize_fix", END)
    return graph.compile()


_compiled_graph = None


def ask(repo_url: str, question: str, file_path: Optional[str] = None) -> dict:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    result = _compiled_graph.invoke({
        "repo_url": repo_url, "question": question, "file_path": file_path,
        "need_issues": False, "need_commits": False, "need_fix": False,
        "issues": [], "commits": [], "contexts": [],
        "proposed_fix": "", "fix_retry_count": 0, "fix_grounded": True,
        "answer": "",
    })
    sources = [
        f"{c['source_path']}" + (f" :: {c['symbol']}" if c["symbol"] else "")
        for c in result["contexts"]
    ]
    sources += [f"Issue #{i['number']}: {i['title']}" for i in result["issues"]]
    sources += [f"Commit {c['sha'][:7]}: {c['commit']['message']}" for c in result["commits"]]
    return {"answer": result["answer"], "sources": sources}