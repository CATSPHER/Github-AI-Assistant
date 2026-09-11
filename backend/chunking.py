"""
V2: parse Python files with the `ast` module and chunk by function/class
boundaries instead of blind character windows. A function or class never
gets split across two chunks anymore (unless it's unusually long, in which
case we sub-split it but keep the symbol name attached to each piece).
Non-Python files fall back to the V1 sliding-window chunker.
"""
import ast
from dataclasses import dataclass
from typing import Optional

from . import config
from .github_loader import RepoFile


@dataclass
class Chunk:
    id: str
    text: str
    source_path: str
    symbol: str  # function/class name, or "" for generic/module-level chunks


def _sliding_window(text: str, path: str, symbol: str, id_prefix: str) -> list[Chunk]:
    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        piece = text[start:start + size]
        header = f"# File: {path}" + (f"\n# {symbol}" if symbol else "")
        chunks.append(Chunk(
            id=f"{id_prefix}::{idx}",
            text=f"{header}\n{piece}",
            source_path=path,
            symbol=symbol,
        ))
        start += size - overlap
        idx += 1
    return chunks


def _chunk_python_ast(file: RepoFile) -> Optional[list[Chunk]]:
    try:
        tree = ast.parse(file.content)
    except SyntaxError:
        return None  # not valid Python (or a syntax our parser version can't handle) — fall back

    lines = file.content.splitlines()
    top_level = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not top_level:
        return None  # e.g. a config/constants-only file — sliding window handles it fine

    chunks: list[Chunk] = []
    covered_lines = set()

    for node in top_level:
        start_line = node.lineno - 1
        if node.decorator_list:
            start_line = min(start_line, node.decorator_list[0].lineno - 1)
        end_line = getattr(node, "end_lineno", node.lineno)
        covered_lines.update(range(start_line, end_line))

        snippet = "\n".join(lines[start_line:end_line])
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        symbol = node.name

        if len(snippet) <= config.CHUNK_SIZE * 1.5:
            text = f"# File: {file.path}\n# {kind}: {symbol}\n{snippet}"
            chunks.append(Chunk(
                id=f"{file.path}::{symbol}",
                text=text,
                source_path=file.path,
                symbol=symbol,
            ))
        else:
            # Long function/class: sub-split but keep the symbol name on every piece
            # so retrieval and citations still point at the right place.
            chunks.extend(_sliding_window(snippet, file.path, f"{kind}: {symbol}", f"{file.path}::{symbol}"))

    # Anything not inside a function/class (imports, constants, top-level setup)
    # becomes its own chunk — useful for "how is X imported/configured" questions.
    remaining_lines = [line for i, line in enumerate(lines) if i not in covered_lines]
    remaining_text = "\n".join(remaining_lines).strip()
    if remaining_text:
        chunks.extend(_sliding_window(remaining_text, file.path, "module-level code", f"{file.path}::module"))

    return chunks


def chunk_file(file: RepoFile) -> list[Chunk]:
    if file.path.endswith(".py"):
        ast_chunks = _chunk_python_ast(file)
        if ast_chunks:
            return ast_chunks
    return _sliding_window(file.content, file.path, "", file.path)


def chunk_files(files: list[RepoFile]) -> list[Chunk]:
    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_file(f))
    return all_chunks