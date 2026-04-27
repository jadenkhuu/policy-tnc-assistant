# Policy Assistant

RAG-based policy Q&A tool for retail support agents. Indexes markdown policy documents locally, retrieves relevant chunks via FAISS vector search, and generates cited answers via Claude Haiku.

---

## How It Works

```
User question
      ↓
Claude rewrites query with policy terminology   (query expansion)
      ↓
Embed query → FAISS vector search → top-10 chunks   (retrieval)
      ↓
Claude answers using only retrieved chunks   (generation)
      ↓
{ answer, citations[] }
```

1. **Ingest** — policy `.md` files are chunked at heading boundaries, embedded with `all-MiniLM-L6-v2`, and stored in a FAISS index. Run once per corpus update.
2. **Retrieve** — at query time, the question is semantically matched against the index to pull the most relevant policy sections.
3. **Generate** — Claude Haiku receives the original question plus retrieved chunks and returns a structured answer with inline citation markers.

---

## Project Structure

```
policy-assistant/
├── backend/
│   ├── app.py            # Flask API — /health, /topics, /query
│   ├── ingest.py         # Parse → chunk → embed → persist FAISS index
│   ├── retriever.py      # Load index, embed query, return top-k chunks
│   ├── llm.py            # Query expansion + answer generation via Claude Haiku
│   ├── requirements.txt
│   ├── .env.example
│   └── data/
│       ├── corpus/       # Drop .md policy files here
│       └── index/        # FAISS index + chunk metadata (gitignored)
└── frontend/
    └── src/
        ├── App.tsx        # Main UI — question form, answer, citations
        └── api.ts         # Typed fetch wrappers for backend API
```

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, Flask |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (flat L2) |
| LLM | Claude Haiku via Anthropic API |

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Anthropic API key](https://console.anthropic.com/)

---

### Backend

#### 1. Create and activate a virtual environment

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows
```

Your terminal prompt will show `(.venv)` when active. Re-activate whenever you open a new terminal.

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=your-api-key-here
CORPUS_PATH=./data/corpus        # optional — defaults to ./data/corpus
```

#### 4. Ingest the corpus

Run once (and again any time policy files are added or updated):

```bash
python ingest.py
```

This reads all `.md` files in `data/corpus/`, chunks them by heading, embeds each chunk, and writes `data/index/faiss.index` + `data/index/chunks.json`.

#### 5. Start the backend

```bash
python app.py
# Listening on http://localhost:5050
```

---

### Frontend

#### 1. Install dependencies

```bash
cd frontend
npm install
```

#### 2. Configure API URL (optional)

By default the frontend talks to `http://localhost:5050`. To point at a different backend, create `frontend/.env.local`:

```env
VITE_API_URL=http://your-backend-host:5050
```

#### 3. Start the dev server

```bash
npm run dev
# http://localhost:5173
```

---

## Usage

Open `http://localhost:5173` and type a question in natural language.

**Example questions:**
- *"A customer wants to return a product but it's been over 20 days."*
- *"Can we price match against an overseas retailer?"*
- *"A customer missed their delivery window — what are their options?"*

Answers include inline citation numbers `(1)`, `(2)` linked to the source policy sections listed below.

---

## Updating the Corpus

1. Add or edit `.md` files in `backend/data/corpus/`
2. Re-run ingestion:
   ```bash
   cd backend && python ingest.py
   ```
3. Restart the backend (index is loaded once at startup):
   ```bash
   python app.py
   ```

Optionally add YAML frontmatter to a policy file to set a canonical title and source URL:

```markdown
---
title: Returns and Refunds Policy
url: https://www.binglee.com.au/returns-policy
---

## Eligibility
...
```

---

## API Reference

### `POST /query`

Submit a question and receive an answer with citations.

```bash
curl -s -X POST http://localhost:5050/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the conditions for a price match?"}' | jq
```

```json
{
  "answer": "To qualify for a price match, the competitor must be an Australian retailer (1) and the item must be identical in brand, model, and colour (2).",
  "citations": [
    { "title": "Price Match Policy", "section": "Price Match Policy > Eligibility", "url": "" },
    { "title": "Price Match Policy", "section": "Price Match Policy > Eligible Items", "url": "" }
  ]
}
```

### `GET /health`

```bash
curl -s http://localhost:5050/health | jq
# { "status": "ok", "chunks_indexed": 142 }
```

### `GET /topics`

Returns all indexed documents with title, source URL, and section count.

```bash
curl -s http://localhost:5050/topics | jq
```

---

## Notes

- Answers are grounded exclusively in the indexed corpus — Claude will not use outside knowledge.
- If retrieved chunks are insufficient to answer, the model says so and advises checking with a supervisor.
- Policy data sourced from Bing Lee's public website. This is a personal portfolio project — no internal systems or customer data involved.
