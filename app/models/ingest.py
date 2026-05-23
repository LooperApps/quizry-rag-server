from typing import Literal
from pydantic import BaseModel, field_validator


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
