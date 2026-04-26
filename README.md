# Policy Assistant

RAG-based policy Q&A tool for retail support agents. Indexes markdown policy documents locally, retrieves relevant chunks, answers via Claude Haiku.

## Stack

- **Backend**: Python/Flask, sentence-transformers, FAISS, Anthropic API
- **Frontend**: React + Vite (Phase 4)

## Project Structure

```
policy-assistant/
├── backend/
│   ├── app.py          # Flask app, /query endpoint
│   ├── ingest.py       # Load, chunk, embed, persist index
│   ├── retriever.py    # Load index, retrieve top-k chunks
│   ├── llm.py          # Generate answer via Claude Haiku
│   ├── requirements.txt
│   ├── .env.example
│   └── data/
│       ├── corpus/     # Drop .md policy files here
│       └── index/      # FAISS index persisted here (gitignored)
└── frontend/           # React app (Phase 4)
```

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
```

## Ingest corpus

```bash
# Drop .md files into backend/data/corpus/, then:
python ingest.py
```

## Run

```bash
python app.py
```

## API

### `POST /query`

```bash
curl -s -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the conditions for a price match?"}' | jq
```

Response:
```json
{
  "answer": "To qualify for a price match, the competitor must be an Australian retailer...",
  "citations": [
    {
      "title": "Price Match Policy",
      "section": "Price Match Policy > Eligibility > Competitor Requirements",
      "url": "https://example.com/policies/price-match"
    }
  ]
}
```

### `GET /health`

```bash
curl -s http://localhost:5000/health | jq
# {"status": "ok", "chunks_indexed": 142}
```
# policy-tnc-assistant
