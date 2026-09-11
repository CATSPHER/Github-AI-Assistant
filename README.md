# AI GitHub Repository Assistant — V1

Ask questions about any GitHub repo: "How does authentication work here?"

## What's actually happening (the concepts)

```
Streamlit UI  →  FastAPI  →  LangGraph (retrieve → generate)  →  HF LLM
                                   ↓
                          Chroma vector store
                                   ↑
                    sentence-transformers embeddings
                                   ↑
                        cloned repo, chunked
```

- **Chunking** (`chunking.py`): a repo's code is too big to hand an LLM whole,
  so it's cut into overlapping ~1200-character pieces small enough to embed
  and retrieve individually.
- **Embeddings** (`vectorstore.py`): each chunk is turned into a vector
  (a list of numbers capturing its meaning) by calling the Hugging Face
  Inference API's feature-extraction task — same `HF_TOKEN`/`InferenceClient`
  the LLM call uses, just a different model and task. No local download, but
  each batch is a network round trip, so indexing is slower than a local
  model would be.
- **Vector store** (Chroma): stores those vectors so that given a *question's*
  embedding, it can find the chunks whose vectors are closest — i.e. the most
  relevant code.
- **RAG** (Retrieval-Augmented Generation): retrieve relevant chunks, stuff
  them into the LLM's prompt as context, and let the LLM answer *grounded in
  your actual code* instead of guessing from training data.
- **LangGraph** (`rag_graph.py`): defines the pipeline as a graph of nodes
  (`retrieve` → `generate`) with typed state flowing between them. Overkill
  for two linear nodes, but this is the exact shape you'll extend with
  branches and loops in V3–V6.

## Setup

```bash
cd repo-assistant
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `HF_TOKEN` to a free token from
https://huggingface.co/settings/tokens (read access is enough).

Some models require you to first accept their license on their HF model page
(Llama models, for example). `Qwen/Qwen2.5-7B-Instruct` in the example env
usually needs no approval — good default to start with.

## Run it

Terminal 1 — API:
```bash
uvicorn backend.main:app --reload --port 8000
```

Terminal 2 — UI:
```bash
streamlit run frontend/app.py
```

Open the Streamlit URL, paste a small public repo URL (start with something
under ~200 files while you're learning — big repos take longer to embed),
click **Ingest repo**, then ask a question.

## If something breaks

- **`InferenceClient` errors about the model** (LLM or embedding one) — the
  model may be gated or temporarily unavailable on HF's free inference; swap
  `HF_LLM_MODEL` or `EMBEDDING_MODEL` in `.env` for another model that
  supports that task and restart the API.
- **Embedding errors mentioning "feature-extraction not supported"** — not
  every model on the Hub serves that task via the free Inference API; stick
  to well-known sentence-transformers model ids (the default one is a safe
  bet), or check the model's page for an "Inference API" widget to confirm.
- **Ingest is slow / rate-limited** — every chunk is embedded over the
  network in small batches with a short delay between them (see
  `EMBED_BATCH_SIZE` / `EMBED_BATCH_DELAY` in `vectorstore.py`); for a large
  repo this can take a while on the free tier. Start with a small repo.
- **Chroma "collection not found" on /ask** — you asked a repo you haven't
  ingested in this session, or the API restarted (in-memory client handle is
  gone but data persists — this rebuilds automatically on next ingest).

## Where to go next (V2 → V6)

1. **V2 — explain specific files/functions**: swap the sliding-window
   chunker for an AST-aware one (`tree-sitter` or Python's `ast` module) so
   chunks respect function/class boundaries, and let `/ask` accept an
   optional `file_path` to scope retrieval.
2. **V3 — pull in GitHub issues**: install `pip install mcp` and connect to
   GitHub's official MCP server (or run `github-mcp-server` locally). Add it
   as a **tool** the LangGraph node can call — this is where LangGraph's
   conditional edges start to matter: "does this question need code, issues,
   or both?"
3. **V4 — analyze recent commits**: another MCP-backed tool (commit log +
   diffs). Add a router node that decides which tool(s) to call before
   generating.
4. **V5 — suggest a fix**: add a `critique` node that checks the proposed
   fix against the retrieved context and loops back to `generate` if it's
   unsupported (your first cyclic graph, not just a straight line).
5. **V6 — human approval → PR**: use LangGraph's `interrupt()` to pause the
   graph before the "create PR" tool call, surface the diff to a human (a
   button in Streamlit), and resume the graph with their approval. This is
   full HITL (human-in-the-loop).

Each step above is a small, additive change to this same codebase — you
don't rebuild anything, you extend the graph and add tools.
