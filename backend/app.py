import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

import llm
import retriever

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174"])

INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")

# Load index once at startup
_chunks_count: int = 0
_index_error: str | None = None

try:
    _chunks_count = retriever.load_index(INDEX_DIR)
    log.info("Index loaded: %d chunks", _chunks_count)
except FileNotFoundError as e:
    _index_error = str(e)
    log.warning("Index not loaded: %s", e)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "chunks_indexed": _chunks_count})


@app.route("/topics", methods=["GET"])
def topics():
    if _index_error:
        return jsonify({"error": _index_error}), 503
    return jsonify(retriever.get_documents())


@app.route("/query", methods=["POST"])
def query():
    if _index_error:
        return jsonify({"error": _index_error}), 503

    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing or empty 'question' field."}), 400

    try:
        retrieval_query = llm.expand_query(question)
        log.info("Expanded query: %s", retrieval_query)
        chunks = retriever.retrieve(retrieval_query, k=10)
        result = llm.answer(question, chunks)
    except Exception as e:
        log.exception("Query failed")
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
