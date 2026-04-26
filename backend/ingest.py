"""
Ingestion pipeline for Policy Assistant.

Chunking strategy
-----------------
Each Markdown file is split at ## and ### heading boundaries. A chunk is
one contiguous block of text beneath a single heading. Each chunk stores:

  heading_path — full breadcrumb from the document title to the immediate
                 heading, e.g. "Price Match Policy > Eligibility > Brands".
                 This gives the LLM provenance without reading the full file.
  body         — raw text content of the section
  title / url  — document-level metadata from YAML frontmatter

For embedding, heading_path + body are concatenated so the vector captures
both the semantic topic and the detailed content. Chunks shorter than
MIN_CHUNK_CHARS characters are merged into the preceding chunk to avoid
indexing near-empty blocks.

Run once (or after any policy update):
    python ingest.py
"""

import hashlib
import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CORPUS_PATH = Path(os.getenv("CORPUS_PATH", "./data/corpus"))
INDEX_DIR = Path("./data/index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
MIN_CHUNK_CHARS = 80  # merge shorter chunks into predecessor


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

# Matches a YAML frontmatter block at the very start of the file
_FRONTMATTER_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata dict, body text with frontmatter removed)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip("\"'")
    return meta, text[m.end():]


def _title_from_path(p: Path) -> str:
    """Derive a human-readable title from a filename."""
    return p.stem.replace("-", " ").replace("_", " ").title()


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents(corpus_path: Path) -> list[dict]:
    """Read all .md files; extract frontmatter title and url."""
    docs = []
    for path in sorted(corpus_path.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        docs.append({
            "title": meta.get("title") or _title_from_path(path),
            "url": meta.get("url", ""),
            "body": body,
            "source": path.name,
        })
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3}) (.+)$")


def _chunk_id(heading_path: str, body_prefix: str) -> str:
    raw = f"{heading_path}|{body_prefix[:64]}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _chunk_document(doc: dict) -> list[dict]:
    """
    Split one document into chunks at ## and ### boundaries.

    Breadcrumb slots: [h1/doc-title, h2, h3].
    H1 headings in the body update breadcrumb[0] but do not flush a chunk —
    they're typically a repeat of the frontmatter title.
    """
    title, url = doc["title"], doc["url"]
    breadcrumb: list[str] = [title, "", ""]
    current_lines: list[str] = []
    chunks: list[dict] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        heading_path = " > ".join(p for p in breadcrumb if p)
        chunks.append({
            "chunk_id": _chunk_id(heading_path, body),
            "heading_path": heading_path,
            "body": body,
            "title": title,
            "url": url,
        })

    for line in doc["body"].splitlines():
        m = _HEADING_RE.match(line)
        if m:
            depth = len(m.group(1))
            text = m.group(2).strip()
            if depth == 1:
                breadcrumb = [text, "", ""]
            elif depth == 2:
                flush()
                current_lines = []
                breadcrumb[1] = text
                breadcrumb[2] = ""
            else:  # depth == 3
                flush()
                current_lines = []
                breadcrumb[2] = text
            continue  # heading line not included in body

        current_lines.append(line)

    flush()

    # Merge undersized chunks into their predecessor
    merged: list[dict] = []
    for chunk in chunks:
        if merged and len(chunk["body"]) < MIN_CHUNK_CHARS:
            merged[-1]["body"] += "\n\n" + chunk["body"]
        else:
            merged.append(chunk)

    return merged


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Chunk all documents; return flat list of chunk dicts."""
    chunks = []
    for doc in docs:
        chunks.extend(_chunk_document(doc))
    return chunks


# ---------------------------------------------------------------------------
# Embedding + FAISS index
# ---------------------------------------------------------------------------

def build_index(chunks: list[dict], model: SentenceTransformer) -> faiss.Index:
    """Embed chunks and return a FAISS flat-L2 index over the embeddings."""
    # Prepend heading path so the vector captures both topic and content
    texts = [f"{c['heading_path']}\n{c['body']}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from {CORPUS_PATH} …")
    docs = load_documents(CORPUS_PATH)
    if not docs:
        print("No .md files found. Drop policy files into data/corpus/ and re-run.")
        return

    print(f"Chunking {len(docs)} document(s) …")
    chunks = chunk_documents(docs)

    print(f"Embedding {len(chunks)} chunks with '{EMBED_MODEL}' …")
    model = SentenceTransformer(EMBED_MODEL)
    index = build_index(chunks, model)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))

    print(f"\nIndexed {len(chunks)} chunks from {len(docs)} files.")
    print(f"  FAISS index → {FAISS_INDEX_PATH}")
    print(f"  Metadata    → {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
