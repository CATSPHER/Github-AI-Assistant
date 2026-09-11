"""
V2 adds an optional file_path scope to the state, and includes each chunk's
symbol (function/class name) in the prompt context so the model can cite
"function foo in bar.py" instead of just the file.
"""
from typing import TypedDict, Optional

from groq import Groq
from langgraph.graph import StateGraph, START, END

from . import config
from .vectorstore import retrieve


class GraphState(TypedDict):
    repo_url: str
    question: str
    file_path: Optional[str]
    contexts: list[dict]
    answer: str


_groq_client = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def retrieve_node(state: GraphState) -> GraphState:
    hits = retrieve(state["repo_url"], state["question"], file_path=state.get("file_path"))
    return {**state, "contexts": hits}


def generate_node(state: GraphState) -> GraphState:
    context_block = "\n\n---\n\n".join(
        f"[{h['source_path']}" + (f" :: {h['symbol']}" if h["symbol"] else "") + f"]\n{h['text']}"
        for h in state["contexts"]
    )
    prompt = (
        "You are a senior engineer explaining a codebase to a teammate. "
        "Answer ONLY using the code context below. If the context doesn't "
        "contain the answer, say so instead of guessing. Cite file paths "
        "(and function/class names when given) in square brackets.\n\n"
        f"CODE CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {state['question']}\n\nANSWER:"
    )

    client = get_groq_client()
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.2,
    )
    answer = completion.choices[0].message.content
    return {**state, "answer": answer}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_compiled_graph = None


def ask(repo_url: str, question: str, file_path: Optional[str] = None) -> dict:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    result = _compiled_graph.invoke({
        "repo_url": repo_url,
        "question": question,
        "file_path": file_path,
        "contexts": [],
        "answer": "",
    })
    return {
        "answer": result["answer"],
        "sources": [
            f"{c['source_path']}" + (f" :: {c['symbol']}" if c["symbol"] else "")
            for c in result["contexts"]
        ],
    }