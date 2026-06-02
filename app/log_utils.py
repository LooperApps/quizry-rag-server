"""
Centralized logging utilities for the Quizry RAG server.

Provides structured, detailed logging with request tracing so every step
of a request can be correlated in the logs.
"""

import logging
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

# ─── Request-scoped trace ID ─────────────────────────────────────────────────

_trace_id: ContextVar[str] = ContextVar("trace_id", default="no-trace")


def get_trace_id() -> str:
    return _trace_id.get()


def set_trace_id(tid: str | None = None) -> str:
    tid = tid or uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


# ─── Formatter helpers ───────────────────────────────────────────────────────

_LINE = "─" * 72


def _fmt(tag: str, msg: str, trace: str = "", **extra: Any) -> str:
    """Build a structured log line with optional extra key=value pairs."""
    parts = [f"[{tag}]"]
    if trace:
        parts.append(f"trace={trace}")
    parts.append(msg)
    for k, v in extra.items():
        parts.append(f"{k}={v!r}")
    return "  ".join(parts)


# ─── Public helpers ───────────────────────────────────────────────────────────

def log_step(logger: logging.Logger, step: str, msg: str, **extra: Any) -> None:
    """Log a pipeline step with trace ID and structured extras."""
    trace = get_trace_id()
    logger.info(_fmt(step, msg, trace=trace, **extra))


def log_prompt(
    logger: logging.Logger,
    caller: str,
    prompt: str,
    response: str | None = None,
    model: str = "",
    **extra: Any,
) -> None:
    """Log a full LLM prompt and its response with clear visual delimiters."""
    trace = get_trace_id()
    header = (
        f"\n{_LINE}\n"
        f"[LLM] trace={trace} caller={caller} model={model}\n"
        f"--- PROMPT ---\n{prompt}\n"
    )
    if response is not None:
        header += f"--- RESPONSE ---\n{response}\n"
    header += _LINE
    logger.info(header)


def log_chroma_query(
    logger: logging.Logger,
    caller: str,
    notebook_ids: list[str],
    query_vector: list[float] | None,
    n_results: int,
    where_filter: dict,
    results: dict[str, Any] | None = None,
) -> None:
    """Log a ChromaDB query with full details."""
    trace = get_trace_id()
    n_dims = len(query_vector) if query_vector else 0
    header = (
        f"\n{_LINE}\n"
        f"[CHROMA] trace={trace} caller={caller}\n"
        f"notebook_ids={notebook_ids!r}\n"
        f"n_results={n_results}\n"
        f"where_filter={where_filter!r}\n"
        f"query_vector_dims={n_dims}\n"
    )
    if results is not None:
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        header += f"result_count={len(docs)}\n"
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            dist = dists[i] if i < len(dists) else "?"
            src = meta.get("source", "?")
            sid = meta.get("sourceId", "?")
            header += (
                f"  result[{i}] distance={dist:.4f} source={src!r} sourceId={sid!r}\n"
                f"    content_preview={doc[:300]!r}\n"
            )
    header += _LINE
    logger.info(header)


def log_step_data(
    logger: logging.Logger,
    step: str,
    label: str,
    data: Any,
    max_len: int = 2000,
) -> None:
    """Log arbitrary step data (e.g. extracted text, embeds, chunk contents)."""
    trace = get_trace_id()
    text = str(data)
    if len(text) > max_len:
        text = text[:max_len] + f"\n... [truncated {len(text) - max_len} more chars]"
    logger.info(
        f"\n{_LINE}\n[DATA:{step}] trace={trace} {label}\n{text}\n{_LINE}"
    )
