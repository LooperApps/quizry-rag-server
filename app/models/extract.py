from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Decoded size limit for inline files (design contract §4).
MAX_BASE64_BYTES = 15 * 1024 * 1024


class ExtractRequest(BaseModel):
    """
    POST /extract — page text of one file, without indexing it. Give exactly
    one of `storageUrl` (downloaded like /ingest) or `fileBase64`.
    """

    fileName: str = Field(min_length=1, max_length=512)
    storageUrl: str | None = None
    fileBase64: str | None = None
    maxVisionPages: int = Field(default=40, ge=0, le=500)

    @model_validator(mode="after")
    def _one_source(self) -> "ExtractRequest":
        if bool(self.storageUrl) == bool(self.fileBase64):
            raise ValueError("Provide exactly one of storageUrl or fileBase64")
        # base64 is 4 chars per 3 bytes; reject before decoding anything huge.
        if self.fileBase64 and len(self.fileBase64) > (MAX_BASE64_BYTES * 4) // 3 + 4:
            raise ValueError(f"fileBase64 exceeds {MAX_BASE64_BYTES // (1024 * 1024)} MB decoded")
        return self


class ExtractedPageOut(BaseModel):
    page: int
    text: str
    method: Literal["text", "ocr", "text+vision"]


class ExtractResponse(BaseModel):
    pageCount: int
    pages: list[ExtractedPageOut]
    visionPages: int
    truncated: bool
