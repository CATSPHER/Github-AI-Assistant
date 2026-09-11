from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .github_loader import clone_repo, load_files, cleanup
from .chunking import chunk_files
from .vectorstore import index_chunks
from .rag_graph import ask as ask_graph

app = FastAPI(title="AI GitHub Repository Assistant")


class IngestRequest(BaseModel):
    repo_url: str


class AskRequest(BaseModel):
    repo_url: str
    question: str
    file_path: Optional[str] = None


@app.post("/ingest")
def ingest(req: IngestRequest):
    """Clone -> chunk -> embed -> store. Call this once per repo before asking questions."""
    try:
        repo_dir = clone_repo(req.repo_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not clone repo: {e}")

    try:
        files = load_files(repo_dir)
        if not files:
            raise HTTPException(status_code=400, detail="No indexable files found in repo.")
        chunks = chunk_files(files)
        collection = index_chunks(req.repo_url, chunks)
    except HTTPException:
        raise
    except Exception as e:
        # Without this, any embedding/indexing failure (bad HF token, model
        # rejecting an oversized input, rate limit, etc.) becomes a bare 500
        # with no detail — this makes the real cause visible in the response.
        raise HTTPException(status_code=500, detail=f"Ingest failed during chunk/embed/store step: {e}")
    finally:
        cleanup(repo_dir)

    return {"status": "indexed", "files": len(files), "chunks": len(chunks), "collection": collection}


@app.post("/ask")
def ask(req: AskRequest):
    """Ask a question about an already-ingested repo, optionally scoped to one file."""
    try:
        result = ask_graph(req.repo_url, req.question, file_path=req.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"{e}. Did you call /ingest for this repo_url first?",
        )
    return result


@app.get("/health")
def health():
    return {"status": "ok"}