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
                    "Add inline citation markers like (1) or (2) immediately after each claim, "
                    "where the number matches the 1-based index of the source in the citations array. "
                    "Example: 'Returns must be within 30 days (1) and include the original receipt (2).' "
                    "If chunks are insufficient, state that clearly and advise the agent "
                    "to check with a supervisor."
                ),
            },
            "citations": {
                "type": "array",
                "description": "Every policy source used in the answer.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title":   {"type": "string", "description": "Document title"},
                        "section": {"type": "string", "description": "Heading path"},
                        "url":     {"type": "string", "description": "Source URL or empty string"},
                    },
                    "required": ["title", "section", "url"],
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
The number must match the 1-based position of that source in the citations array you return. \
Example: "The change-of-mind window is 14 days (1) and proof of purchase is required (2)."
3. Keep answers to 2–4 sentences — agents are in live chat with customers.
4. If the provided chunks do not contain enough information to answer the question, \
say so explicitly and tell the agent to check with a supervisor.
5. Always use the answer_query tool to return your response.\
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
    return tool_block.input  # already a dict matching our schema
