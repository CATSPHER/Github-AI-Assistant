"""
Embeddings come from the HF Inference API (feature-extraction task) — same
InferenceClient the LLM call uses, no local model download. This is slower
per-chunk than a local model since each request is a network round trip, so
indexing is done in small batches with a short delay to stay under free-tier
rate limits.
Chroma stores vectors on disk in config.CHROMA_DIR, keyed by collection name
so multiple repos can live side by side.
"""
import time
from typing import Optional

import chromadb
from huggingface_hub import InferenceClient

from . import config
from .chunking import Chunk

_hf_embed_client = None
_client = None

EMBED_BATCH_SIZE = 16     # texts per HF API call
EMBED_BATCH_DELAY = 0.2   # seconds between batches, to be polite to the free tier


def get_hf_embed_client() -> InferenceClient:
    global _hf_embed_client
    if _hf_embed_client is None:
        _hf_embed_client = InferenceClient(provider="hf-inference", token=config.HF_TOKEN, model=config.EMBEDDING_MODEL)
    return _hf_embed_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a list of texts via the HF Inference API, batched."""
    client = get_hf_embed_client()
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        result = client.feature_extraction(batch)
        vectors = result.tolist() if hasattr(result, "tolist") else result

        pooled = []
        for vec in vectors:
            if isinstance(vec[0], list):
                pooled.append([sum(dim) / len(dim) for dim in zip(*vec)])
            else:
                pooled.append(vec)
        all_vectors.extend(pooled)
        time.sleep(EMBED_BATCH_DELAY)
    return all_vectors


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client


def collection_name(repo_url: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in repo_url)[:60]


def index_chunks(repo_url: str, chunks: list[Chunk]) -> str:
    name = collection_name(repo_url)
    client = get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name)

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source_path": c.source_path, "symbol": c.symbol} for c in chunks],
    )
    return name


def retrieve(repo_url: str, query: str, top_k: int = config.TOP_K, file_path: Optional[str] = None) -> list[dict]:
    client = get_client()
    collection = client.get_collection(collection_name(repo_url))
    query_embedding = embed_texts([query])

    if file_path:
        raw = collection.query(query_embeddings=query_embedding, n_results=max(top_k * 8, 50))
        hits = []
        for text, meta in zip(raw["documents"][0], raw["metadatas"][0]):
            if file_path in meta["source_path"]:
                hits.append({"text": text, "source_path": meta["source_path"], "symbol": meta.get("symbol", "")})
            if len(hits) >= top_k:
                break
        return hits

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    hits = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"text": text, "source_path": meta["source_path"], "symbol": meta.get("symbol", "")})
    return hits