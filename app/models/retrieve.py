from pydantic import BaseModel, Field, model_validator


class RetrieveRequest(BaseModel):
    # Provide either notebookId (single notebook) or notebookIds (multi-notebook
    # research feature). notebookId is a convenience alias for notebookIds=[notebookId].
    notebookId: str | None = None
    notebookIds: list[str] | None = None

    question: str = Field(min_length=1, max_length=4096)
    k: int = Field(default=10, ge=1, le=20)

    # Optional recent chat turns ({"role": "user"|"assistant", "content": str})
    # used for context-aware query rewriting of follow-up questions.
    history: list[dict] | None = None

    @model_validator(mode="after")
    def resolve_ids(self) -> "RetrieveRequest":
        if self.notebookIds and self.notebookId:
            # notebookIds takes precedence
            return self
        if self.notebookId and not self.notebookIds:
            self.notebookIds = [self.notebookId]
        if not self.notebookIds:
            raise ValueError("Provide notebookId or notebookIds")
        # Deduplicate and filter empty strings
        self.notebookIds = list(dict.fromkeys(nid for nid in self.notebookIds if nid.strip()))
        if not self.notebookIds:
            raise ValueError("notebookIds must contain at least one non-empty value")
        return self


class ChunkResult(BaseModel):
    content: str
    source: str       # original file name
    sourceId: str     # Firestore source document ID


class RetrieveResponse(BaseModel):
    chunks: list[ChunkResult]
    rewrittenQuery: str | None = None
