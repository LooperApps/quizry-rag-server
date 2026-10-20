"""
Centralized logging utilities for the Quizry RAG server.

Provides structured, detailed logging with request tracing so every step
of a request can be correlated in the logs.
"""

import json
import logging
import re
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


# ─── Redaction ───────────────────────────────────────────────────────────────

# Firebase download URLs carry a bearer-equivalent `token=` query parameter;
# signed URLs carry signatures. None of it belongs in a log line.
_SECRET_PARAM = re.compile(
    r"((?:[?&]|\\u0026)(?:token|access_token|key|sig|signature|x-goog-signature)=)[^&#\"'\s\\]+",
    re.IGNORECASE,
)
_BASE64_FIELD = re.compile(r'("fileBase64"\s*:\s*")([^"]*)(")')
BODY_LOG_CHARS = 2000


def redact_url(url: str) -> str:
    """The URL with token/signature query values replaced by <redacted>."""
    return _SECRET_PARAM.sub(r"\1<redacted>", url or "")


def redact_body(path: str, body: str, limit: int = BODY_LOG_CHARS) -> str:
    """
    A request/response body made safe and short for the log: download tokens
    redacted, base64 payloads replaced by their length, page text summarised
    (POST /ingest/text bodies are whole books), and the rest truncated.
    """
    if path.rstrip("/").endswith("/ingest/text"):
        try:
            data = json.loads(body)
            pages = data.get("pages") or []
            return json.dumps({
                "notebookId": data.get("notebookId"),
                "sourceId": data.get("sourceId"),
                "fileName": data.get("fileName"),
                "collection": data.get("collection"),
                "pages": len(pages),
                "chars": sum(len(str(p.get("text") or "")) for p in pages if isinstance(p, dict)),
            }, ensure_ascii=False)
        except (ValueError, AttributeError, TypeError):
            pass
    body = _BASE64_FIELD.sub(lambda m: f"{m.group(1)}<{len(m.group(2))} base64 chars>{m.group(3)}", body)
    body = redact_url(body)
    if len(body) > limit:
        body = body[:limit] + f"\n... [truncated {len(body) - limit} more chars]"
    return body


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
