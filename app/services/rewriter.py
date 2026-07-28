"""
Query rewriting service.

Rewrites the user's natural-language question into a concise, keyword-dense
search query that is more likely to surface relevant content in ChromaDB.
"""

import logging
import time

from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.log_utils import get_trace_id, log_prompt, log_step

logger = logging.getLogger(__name__)
LINE = "=" * 80


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
    trace = get_trace_id()
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
        "The knowledge base may be written in Arabic, Hebrew, or English — the student's "
        "study material can be in any of these. Keep the rewritten query in the SAME language "
        "as the question, then append translations of the 2-3 most important domain terms in "
        "the other likely languages (Hebrew and English for Arabic questions; Arabic and "
        "English for Hebrew questions) so the query matches the material regardless of its "
        "language.\n\n"
        "If the question is a follow-up (e.g. 'explain more', 'give me an example'), use the "
        "recent conversation to resolve what topic it refers to and search for that topic.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Student question: {question}\n\n"
        "Return ONLY the rewritten query — no explanation, no punctuation around it."
    )

    t0 = time.perf_counter()
    log_step(
        logger, "REWRITER",
        f"calling model={settings.litellm_model} question={question!r}",
    )

    response = completion(
        model=settings.litellm_model,
        messages=[{"role": "user", "content": prompt}],
    )
    rewritten = response.choices[0].message.content.strip()
    elapsed = int((time.perf_counter() - t0) * 1000)

    log_prompt(
        logger, "rewrite_query",
        prompt=prompt,
        response=rewritten,
        model=settings.litellm_model,
        elapsed_ms=elapsed,
    )

    log_step(
        logger, "REWRITER",
        f"done elapsed={elapsed}ms result={rewritten!r}",
    )
    return rewritten
