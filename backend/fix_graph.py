"""
V6: propose-a-fix workflow with a REAL human-in-the-loop pause, unlike V5
which just returned a diff for you to read. This graph actually stops mid-
execution using LangGraph's interrupt() and only continues to create a
branch, push files, and open a PR once a human calls back with a decision.

This requires a checkpointer (MemorySaver here -- state lives in the running
Python process's memory, so it's lost on restart; swapping to a persistent
one like SqliteSaver is a natural next step, not done here to keep this
version simple to reason about).
"""
import ast
import difflib
import uuid
from typing import TypedDict, Optional

from groq import Groq
from langgraph.graph import StateGraph, START, END
try:
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver
    
from langgraph.types import Command, interrupt

from . import config
from .github_tools import get_file_contents, parse_owner_repo, create_branch, push_files, create_pull_request


class FixState(TypedDict):
    repo_url: str
    target_file: str
    fix_request: str
    original_content: str
    proposed_content: str
    diff_text: str
    valid_syntax: bool
    approved: Optional[bool]
    branch_name: str
    pr_url: str
    answer: str


_groq_client = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def fetch_original_node(state: FixState) -> FixState:
    content = get_file_contents(state["repo_url"], state["target_file"])
    return {**state, "original_content": content}


def generate_fix_node(state: FixState) -> FixState:
    prompt = (
        "You are a senior engineer fixing a bug. Below is the CURRENT FULL "
        "content of one file. Rewrite the ENTIRE file with the requested fix "
        "applied. Output ONLY the complete corrected file content — no "
        "explanation, no markdown code fences. This output will be written "
        "directly as the new file content.\n\n"
        f"FILE: {state['target_file']}\n\n"
        f"CURRENT CONTENT:\n{state['original_content']}\n\n"
        f"REQUESTED FIX: {state['fix_request']}\n\n"
        "CORRECTED FILE CONTENT:"
    )
    client = get_groq_client()
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=2000, temperature=0.1,
    )
    proposed = completion.choices[0].message.content.strip()
    if proposed.startswith("```"):
        lines = proposed.split("\n")
        proposed = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return {**state, "proposed_content": proposed}


def validate_node(state: FixState) -> FixState:
    valid = True
    if state["target_file"].endswith(".py"):
        try:
            ast.parse(state["proposed_content"])
        except SyntaxError:
            valid = False

    diff = difflib.unified_diff(
        state["original_content"].splitlines(keepends=True),
        state["proposed_content"].splitlines(keepends=True),
        fromfile=f"a/{state['target_file']}",
        tofile=f"b/{state['target_file']}",
    )
    return {**state, "valid_syntax": valid, "diff_text": "".join(diff)}


def human_review_node(state: FixState) -> FixState:
    # Graph execution actually pauses here. The dict passed to interrupt()
    # is what the API caller receives while paused; whatever gets passed
    # back via Command(resume=...) becomes this call's return value.
    decision = interrupt({
        "diff": state["diff_text"],
        "valid_syntax": state["valid_syntax"],
        "target_file": state["target_file"],
    })
    return {**state, "approved": decision.get("approved", False)}


def after_review(state: FixState) -> str:
    return "apply_fix" if state["approved"] else "cancelled"


def apply_fix_node(state: FixState) -> FixState:
    branch_name = f"ai-fix-{uuid.uuid4().hex[:8]}"
    create_branch(state["repo_url"], branch_name)
    push_files(
        state["repo_url"], branch=branch_name,
        files=[{"path": state["target_file"], "content": state["proposed_content"]}],
        message=f"AI-suggested fix: {state['fix_request'][:72]}",
    )
    pr_url = create_pull_request(
        state["repo_url"],
        title=f"AI-suggested fix: {state['fix_request'][:72]}",
        body=f"Automated fix proposed by the AI GitHub Repository Assistant.\n\n"
             f"Request: {state['fix_request']}\n\n```diff\n{state['diff_text']}\n```",
        head=branch_name, base="main",
    )
    return {**state, "branch_name": branch_name, "pr_url": pr_url, "answer": f"PR created: {pr_url}"}


def cancelled_node(state: FixState) -> FixState:
    return {**state, "answer": "Fix was rejected — no changes were made to the repository."}


def build_fix_graph():
    graph = StateGraph(FixState)
    graph.add_node("fetch_original", fetch_original_node)
    graph.add_node("generate_fix", generate_fix_node)
    graph.add_node("validate", validate_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("apply_fix", apply_fix_node)
    graph.add_node("cancelled", cancelled_node)

    graph.add_edge(START, "fetch_original")
    graph.add_edge("fetch_original", "generate_fix")
    graph.add_edge("generate_fix", "validate")
    graph.add_edge("validate", "human_review")
    graph.add_conditional_edges("human_review", after_review, {
        "apply_fix": "apply_fix", "cancelled": "cancelled",
    })
    graph.add_edge("apply_fix", END)
    graph.add_edge("cancelled", END)

    return graph.compile(checkpointer=MemorySaver())


_fix_graph = None


def get_fix_graph():
    global _fix_graph
    if _fix_graph is None:
        _fix_graph = build_fix_graph()
    return _fix_graph


def propose_fix(repo_url: str, target_file: str, fix_request: str) -> dict:
    graph = get_fix_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({
        "repo_url": repo_url, "target_file": target_file, "fix_request": fix_request,
        "original_content": "", "proposed_content": "", "diff_text": "",
        "valid_syntax": True, "approved": None, "branch_name": "", "pr_url": "", "answer": "",
    }, config=cfg)

    interrupt_info = result.get("__interrupt__")
    if interrupt_info:
        payload = interrupt_info[0].value
        return {"thread_id": thread_id, "status": "awaiting_approval", **payload}
    return {"thread_id": thread_id, "status": "done", "answer": result.get("answer", "")}


def resume_fix(thread_id: str, approved: bool) -> dict:
    graph = get_fix_graph()
    cfg = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume={"approved": approved}), config=cfg)
    return {"status": "done", "answer": result.get("answer", ""), "pr_url": result.get("pr_url", "")}