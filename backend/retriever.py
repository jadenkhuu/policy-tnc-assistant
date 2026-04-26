"""Load persisted FAISS index and retrieve top-k chunks for a query."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"

# Module-level state; populated once by load_index()
_index: faiss.Index | None = None
_chunks: list[dict] = []
_model: SentenceTransformer | None = None


def load_index(index_dir: str | Path) -> int:
    """
    Load FAISS index, chunk metadata, and embedding model into memory.
    Returns the number of indexed chunks.
    Raises FileNotFoundError if index files are missing.
    """
    global _index, _chunks, _model

    index_dir = Path(index_dir)
    faiss_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.json"

    if not faiss_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_dir}. Run `python ingest.py` first."
        )

    _index = faiss.read_index(str(faiss_path))
    _chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    _model = SentenceTransformer(EMBED_MODEL)

    return len(_chunks)


def get_documents() -> list[dict]:
    """Return one entry per unique source document derived from loaded chunks."""
    seen: dict[str, dict] = {}
    for chunk in _chunks:
        title = chunk["title"]
        if title not in seen:
            seen[title] = {"title": title, "url": chunk.get("url", ""), "section_count": 0}
        seen[title]["section_count"] += 1
    return sorted(seen.values(), key=lambda d: d["title"])


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed query and return top-k chunks sorted by L2 distance (ascending).
    Each returned dict adds a 'distance' field to the stored chunk metadata.
    """
    if _index is None or _model is None:
        raise RuntimeError("Index not loaded. Call load_index() first.")

    embedding = _model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = _index.search(embedding, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(_chunks[idx])
        chunk["distance"] = float(dist)
        results.append(chunk)

    return results
