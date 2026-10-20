from pydantic import BaseModel, Field, model_validator

MAX_QUERIES = 12


def _clean_ids(values: list[str] | None) -> list[str]:
    """Strip, drop empties, deduplicate (order kept)."""
    return list(dict.fromkeys(v.strip() for v in (values or []) if v and v.strip()))


class RetrieveRequest(BaseModel):
    # Provide either notebookId (single notebook) or notebookIds (multi-notebook
    # research feature). notebookId is a convenience alias for notebookIds=[notebookId].
    notebookId: str | None = None
    notebookIds: list[str] | None = None
    # Chunks of these sources match too (OR-ed with the notebooks): sources
    # whose chunks are stored under another notebook (shared library books).
    sourceIds: list[str] | None = None

    # Required unless `queries` is given.
    question: str | None = Field(default=None, min_length=1, max_length=4096)
    # Multi-query retrieval: every query is embedded in one batch and searched
    # in one Chroma call; results are merged round-robin and deduplicated. No
    # rewrite, no rerank. `question`, when also given, is the first query.
    queries: list[str] | None = Field(default=None, max_length=MAX_QUERIES)
    # Single-question mode only: false skips the LLM query rewrite.
    rewrite: bool | None = None
    k: int = Field(default=10, ge=1, le=40)

    # Optional recent chat turns ({"role": "user"|"assistant", "content": str})
    # used for context-aware query rewriting of follow-up questions.
    history: list[dict] | None = None

    # The app's live collection when omitted; otherwise one of its
    # `<live>__<suffix>` staging collections (see app/apps.py).
    collection: str | None = None

    @model_validator(mode="after")
    def resolve_ids(self) -> "RetrieveRequest":
        # notebookIds takes precedence over notebookId when both are given.
        ids = self.notebookIds if self.notebookIds else ([self.notebookId] if self.notebookId else [])
        self.notebookIds = _clean_ids(ids)
        self.sourceIds = _clean_ids(self.sourceIds)
        if not self.notebookIds and not self.sourceIds:
            raise ValueError("Provide notebookId, notebookIds or sourceIds")

        # An empty (or all-blank) `queries` list means single-question mode.
        queries = [q.strip()[:4096] for q in (self.queries or []) if q and q.strip()]
        if queries:
            if self.question and self.question.strip():
                queries.insert(0, self.question.strip())
            self.queries = list(dict.fromkeys(queries))[:MAX_QUERIES]
        else:
            self.queries = None
            if not self.question:
                raise ValueError("Provide question or queries")
        return self


class ChunkResult(BaseModel):
    content: str
    source: str       # original file name
    sourceId: str     # Firestore source document ID
    notebookId: str | None = None   # notebook the chunk is stored under
    pageStart: int | None = None    # page span (pages_v1 collections only)
    pageEnd: int | None = None
    distance: float | None = None   # Chroma distance (lower is closer)


class RetrieveResponse(BaseModel):
    chunks: list[ChunkResult]
    rewrittenQuery: str | None = None
