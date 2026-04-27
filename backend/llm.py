"""Generate structured answers via Claude Haiku given retrieved policy chunks."""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


# Tool definition forces Claude to return structured JSON-compatible output
_ANSWER_TOOL = {
    "name": "answer_query",
    "description": "Return a structured answer with citations drawn from policy chunks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Concise answer in 2–4 sentences, based only on the provided chunks. "
                    "Add inline citation markers like (1) or (2) immediately after each claim. "
                    "Number them sequentially in order of first appearance: the first new source you cite = (1), "
                    "second new source = (2), etc. Each number must match the 1-based position of that "
                    "source in the citations array. "
                    "Example: 'Returns must be within 30 days (1) and include the original receipt (2).' "
                    "If chunks are insufficient, state that clearly and advise the agent "
                    "to check with a supervisor — omit citations entirely in that case."
                ),
            },
            "citations": {
                "type": "array",
                "description": (
                    "Sources cited in the answer, in the order they are first cited. "
                    "citations[0] = source (1), citations[1] = source (2), etc."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_index": {
                            "type": "integer",
                            "description": (
                                "The [N] number of the context chunk this citation draws from (1-based). "
                                "Must exactly match one of the [1], [2], ... numbers in the provided policy context."
                            ),
                        },
                        "title":   {"type": "string", "description": "Document title from the chunk's Source line"},
                        "section": {"type": "string", "description": "Heading path from the chunk"},
                        "url":     {"type": "string", "description": "Source URL from the chunk, or empty string"},
                    },
                    "required": ["chunk_index", "title", "section", "url"],
                },
            },
        },
        "required": ["answer", "citations"],
    },
}

# System prompt is static — eligible for prompt caching
_SYSTEM_PROMPT = """\
You are a policy assistant for retail customer support agents. Answer agent questions about \
company policies accurately and concisely.

Rules:
1. Answer ONLY from the policy chunks provided. Do not use outside knowledge.
2. Add inline citation markers — (1), (2), etc. — immediately after each factual claim. \
Number them sequentially in order of first appearance: the first new context chunk you cite \
becomes (1), the next new chunk (2), and so on. Each number must match the 1-based index of \
that source in the citations array. In citations, set chunk_index to the [N] number of the \
context chunk (e.g. chunk_index: 3 if you drew from [3]). \
Example: "The change-of-mind window is 14 days (1) and proof of purchase is required (2)." \
If no chunks support a claim, omit the citation — never invent one.
3. Keep answers to 2–4 sentences — agents are in live chat with customers.
4. If the provided chunks do not contain enough information to answer the question, \
say so explicitly and tell the agent to check with a supervisor.
5. IMPORTANT — never give a confident "yes" to eligibility questions (price match, returns, \
delivery, promotions) unless the chunks explicitly confirm ALL conditions are met AND no \
exclusions apply to the specific case. If the general policy conditions seem met but you \
cannot confirm the specific competitor, brand, or product is not excluded, say the general \
conditions appear to be met but advise the agent to verify at the time of matching or check \
with a supervisor before confirming to the customer.
6. Always use the answer_query tool to return your response.\
"""


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {chunk['heading_path']}\n"
            f"Source: {chunk['title']}"
            + (f" ({chunk['url']})" if chunk.get("url") else "")
            + f"\n\n{chunk['body']}"
        )
    return "\n\n---\n\n".join(parts)


def expand_query(query: str) -> str:
    """
    Rewrite the user query into a retrieval-optimised form.
    Adds policy terminology (exclusions, restrictions, eligibility) so FAISS
    scores specific exception clauses more highly.
    """
    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": (
                "You are a search query optimizer for a retail policy knowledge base. "
                "Rewrite the question below into a search query that will best retrieve "
                "relevant policy text — include terms like 'excluded', 'restrictions', "
                "'conditions', 'eligibility', 'cannot', 'not eligible' where appropriate. "
                "Return only the rewritten query, no explanation.\n\n"
                f"Question: {query}"
            ),
        }],
    )
    return response.content[0].text.strip()


def answer(query: str, chunks: list[dict]) -> dict:
    """
    Call Claude Haiku with the query and retrieved chunks.
    Returns {"answer": str, "citations": [{"title", "section", "url"}]}.
    Raises anthropic.APIError on API failure.
    """
    client = _get_client()
    context = _format_context(chunks)
    user_message = f"Question: {query}\n\nPolicy context:\n\n{context}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=768,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache static system prompt
            }
        ],
        tools=[_ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "answer_query"},  # force tool use
        messages=[{"role": "user", "content": user_message}],
    )

    # Extract the tool_use block — guaranteed present due to tool_choice
    tool_block = next(b for b in response.content if b.type == "tool_use")
    result = tool_block.input

    # Override citation metadata from actual retrieved chunks so hallucinated
    # titles/sections/urls are impossible — chunk_index is the ground truth.
    validated = []
    for citation in result.get("citations", []):
        idx = citation.get("chunk_index", 0) - 1  # convert to 0-based
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            validated.append({
                "chunk_index": idx + 1,
                "title": chunk["title"],
                "section": chunk["heading_path"],
                "url": chunk.get("url", ""),
            })
    result["citations"] = validated
    return result
