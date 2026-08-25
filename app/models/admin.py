from pydantic import BaseModel, Field, model_validator


class AdminQueryRequest(BaseModel):
    """
    A traced retrieval. Every pipeline stage is toggleable so a disappointing
    answer can be bisected: turn rewrite off to see the raw vectors, turn
    rerank off to see Chroma's own ordering.
    """

    question: str = Field(min_length=1, max_length=4096)

    # Omit both to search the whole collection - the manager's "search
    # everywhere" mode, which /retrieve deliberately does not allow.
    notebookId: str | None = None
    notebookIds: list[str] | None = None
    # Narrow to a single file inside the notebook.
    sourceId: str | None = None

    k: int = Field(default=20, ge=1, le=200, description="Candidates pulled from Chroma")
    finalK: int | None = Field(default=None, ge=1, le=200, description="Trim after reranking")

    rewrite: bool = False
    rerank: bool = False
    history: list[dict] | None = None

    @model_validator(mode="after")
    def _clean(self) -> "AdminQueryRequest":
        ids = list(self.notebookIds or [])
        if self.notebookId:
            ids.append(self.notebookId)
        self.notebookIds = list(dict.fromkeys(i.strip() for i in ids if i and i.strip()))
        return self

    def resolved_notebook_ids(self) -> list[str]:
        return self.notebookIds or []


class DeleteChunksRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=1000)


class DeleteResponse(BaseModel):
    status: str
    deleted: int
    notebookId: str | None = None
    sourceId: str | None = None
    detail: str | None = None
