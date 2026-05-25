"""
Query rewriting service.

Rewrites the user's natural-language question into a concise, keyword-dense
search query that is more likely to surface relevant content in ChromaDB.
"""

import logging

from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)


@retry(
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(2),
    reraise=True,
)
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """
    Rewrite `question` into an optimised knowledge-base search query.

    `history` is an optional list of {"role": ..., "content": ...} dicts
    representing recent conversation turns (used for context-aware rewrites).
    Returns the rewritten query string.
    """
    # Include the last 6 turns at most to stay within token limits
    recent = (history or [])[-6:]
    history_text = (
        "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
        if recent
        else "(no prior conversation)"
    )

    prompt = (
        "You are searching an educational Knowledge Base on behalf of a student.\n\n"
        "Rewrite the student's question into a short, precise search query that will surface "
        "the most relevant content. Focus on key concepts, terms, and specifics. "
        "Remove conversational filler. Preserve all domain-specific vocabulary.\n\n"
        "IMPORTANT: This knowledge base is written in Hebrew. "
        "If the question is in Arabic or contains Arabic words referring to Hebrew concepts "
        "(e.g. Israeli universities, academic subjects), translate the key terms to Hebrew "
        "in the rewritten query so it matches the Hebrew content in the knowledge base.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Student question: {question}\n\n"
        "Return ONLY the rewritten query — no explanation, no punctuation around it."
    )

    response = completion(
        model=settings.litellm_model,
        messages=[{"role": "user", "content": prompt}],
    )
    rewritten = response.choices[0].message.content.strip()
    logger.debug(f"[rewriter] '{question}' → '{rewritten}'")
    return rewritten
