from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class IngestRequest(BaseModel):
    notebookId: str
    sourceId: str
    storageUrl: str
    fileName: str
    # fileType is inferred from fileName extension — this field is advisory only
    fileType: Literal["pdf", "text", "docx", "pptx", "html", "md", "unknown"] = "unknown"

    @field_validator("notebookId", "sourceId")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Must not be empty")
        return v.strip()


class IngestResponse(BaseModel):
    status: Literal["ok", "error"]
    sourceId: str
    chunkCount: int = 0
    error: str | None = None


# A whole book is one request; keep a ceiling well above the largest textbook
# (~3k pages, ~8M chars) but below what would exhaust the instance.
MAX_TEXT_PAGES = 5000
MAX_TEXT_CHARS = 20_000_000


class PageText(BaseModel):
    page: int = Field(ge=1)
    text: str


class IngestTextRequest(BaseModel):
    """
    POST /ingest/text — page text extracted elsewhere (the library pipeline).
    `collection` may name the app's live collection or one of its
    `<live>__<suffix>` staging collections (see app/apps.py); omitted means live.
    """

    notebookId: str
    sourceId: str
    fileName: str = Field(min_length=1, max_length=512)
    pages: list[PageText] = Field(min_length=1, max_length=MAX_TEXT_PAGES)
    collection: str | None = None

    @field_validator("notebookId", "sourceId")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def _size(self) -> "IngestTextRequest":
        total = sum(len(p.text) for p in self.pages)
        if total > MAX_TEXT_CHARS:
            raise ValueError(f"pages hold {total:,} chars; the limit is {MAX_TEXT_CHARS:,}")
        if not any(p.text.strip() for p in self.pages):
            raise ValueError("Every page is empty")
        return self


class IngestTextResponse(BaseModel):
    status: Literal["ok"]
    sourceId: str
    chunkCount: int
    pageCount: int
    collection: str
