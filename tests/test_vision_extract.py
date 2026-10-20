"""
DeepSeek vision request shape (HTTP mocked), image preparation, and page-wise
extraction of PDFs built in the test (vision calls mocked). No network.
"""

import base64
import io
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from tenacity import wait_none

from app.main import app
from app.services import extractor, vision

client = TestClient(app)
DAFTAR = {"Authorization": "Bearer test-daftar-key"}

ARABIC = "الدرس الأول مفهوم العدد النسبي العدد النسبي هو كل عدد يمكن كتابته على صورة كسر"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _png(size=(300, 200), mode="RGB", fmt="PNG") -> bytes:
    img = Image.new(mode, size, "white")
    draw = ImageDraw.Draw(img)
    for i in range(0, size[0], 20):
        draw.rectangle([i, 20, i + 10, size[1] - 20], fill="black")
    out = io.BytesIO()
    img.save(out, format=fmt)
    return out.getvalue()


def _text_pdf(lines_per_page: list[list[str]]) -> bytes:
    """A minimal PDF with Helvetica text, one page per entry."""
    objects: list[bytes] = []
    n = len(lines_per_page)
    page_ids = [3 + 2 * i for i in range(n)]
    font_id = 3 + 2 * n
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids).encode()
    objects.append(b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(n).encode() + b" >>")
    for i, lines in enumerate(lines_per_page):
        ops = [b"BT /F1 11 Tf 50 750 Td 14 TL"]
        for line in lines:
            ops.append(b"(" + line.encode("latin-1") + b") Tj T*")
        ops.append(b"ET")
        stream = b"\n".join(ops)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 " + str(font_id).encode() + b" 0 R >> >> "
            b"/Contents " + str(page_ids[i] + 1).encode() + b" 0 R >>"
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(f"{num} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def _image_pdf() -> bytes:
    """A 'scanned' page: one full-page image, no text layer."""
    img = Image.open(io.BytesIO(_png((850, 1100)))).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PDF", resolution=100)
    return out.getvalue()


def _merge(*pdfs: bytes) -> bytes:
    import pypdf

    writer = pypdf.PdfWriter()
    for data in pdfs:
        for page in pypdf.PdfReader(io.BytesIO(data)).pages:
            writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _overlay(text_pdf: bytes, image_pdf: bytes) -> bytes:
    """Text page with a large figure drawn over it."""
    import pypdf

    writer = pypdf.PdfWriter(clone_from=io.BytesIO(text_pdf))
    figure = pypdf.PdfReader(io.BytesIO(image_pdf)).pages[0]
    writer.pages[0].merge_transformed_page(figure, pypdf.Transformation().scale(0.4).translate(100, 100))
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


ENGLISH_PAGE = [f"Line {i}: photosynthesis converts light energy into chemical energy." for i in range(25)]


# ─── vision request shape ─────────────────────────────────────────────────────

class TestVisionRequest:
    @pytest.fixture
    def transport(self, monkeypatch):
        seen: list[httpx.Request] = []
        replies: list[httpx.Response] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if replies:
                return replies.pop(0)
            return httpx.Response(200, json={
                "model": "deepseek-flash",
                "choices": [{"message": {"content": "  نص الصفحة  "}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1100, "completion_tokens": 600},
            })

        monkeypatch.setattr(vision, "_client", httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://api.deepseek.com"))
        monkeypatch.setattr(vision, "_post", vision._post.retry_with(wait=wait_none()))
        return seen, replies

    def test_body_shape(self, transport):
        seen, _ = transport
        result = vision.ocr_image(_png(), label="t")
        assert result.text == "نص الصفحة"
        assert result.prompt_tokens == 1100 and result.completion_tokens == 600
        req = seen[0]
        assert req.url.path == "/chat/completions"
        assert req.headers["authorization"] == "Bearer test-deepseek-key"
        body = json.loads(req.content)
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        assert body["temperature"] == 0 and body["max_tokens"] == 3000
        (message,) = body["messages"]
        assert message["role"] == "user"
        text_part, image_part = message["content"]
        assert text_part == {"type": "text", "text": vision.OCR_PROMPT}
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_describe_prompt(self, transport):
        seen, _ = transport
        vision.describe_image(_png())
        body = json.loads(seen[0].content)
        assert body["messages"][0]["content"][0]["text"] == vision.DESCRIBE_PROMPT

    def test_retries_5xx_then_succeeds(self, transport):
        seen, replies = transport
        replies.extend([httpx.Response(503, text="busy"), httpx.Response(429, text="slow down")])
        assert vision.ocr_image(_png()).text == "نص الصفحة"
        assert len(seen) == 3

    def test_gives_up_after_four_attempts(self, transport):
        seen, replies = transport
        replies.extend([httpx.Response(500, text="x")] * 4)
        with pytest.raises(vision.VisionError):
            vision.ocr_image(_png())
        assert len(seen) == 4

    def test_4xx_is_not_retried(self, transport):
        seen, replies = transport
        replies.append(httpx.Response(400, text="bad image"))
        with pytest.raises(vision.VisionError):
            vision.ocr_image(_png())
        assert len(seen) == 1

    def test_key_never_in_error(self, transport):
        _, replies = transport
        replies.append(httpx.Response(401, text="unauthorized"))
        with pytest.raises(vision.VisionError) as info:
            vision.ocr_image(_png())
        assert "test-deepseek-key" not in str(info.value)


class TestPrepareImage:
    def test_sniff(self):
        assert vision.sniff_mime(_png()) == "image/png"
        assert vision.sniff_mime(_png(fmt="GIF")) == "image/gif"
        assert vision.sniff_mime(_png(fmt="JPEG")) == "image/jpeg"
        assert vision.sniff_mime(_png(fmt="BMP")) == "image/bmp"
        assert vision.sniff_mime(b"not an image") is None

    def test_supported_passthrough(self):
        data = _png()
        assert vision.prepare_image(data) == (data, "image/png")

    def test_bmp_and_tiff_converted_to_jpeg(self):
        for fmt in ("BMP", "TIFF"):
            payload, mime = vision.prepare_image(_png(fmt=fmt))
            assert mime == "image/jpeg" and payload[:3] == b"\xff\xd8\xff"

    def test_rgba_flattened_and_downscaled(self, monkeypatch):
        monkeypatch.setattr(vision, "MAX_SIDE_PX", 100)
        payload, mime = vision.prepare_image(_png((400, 50), mode="RGBA"))
        assert mime == "image/jpeg"
        assert max(Image.open(io.BytesIO(payload)).size) <= 100

    def test_garbage_rejected(self):
        with pytest.raises(vision.UnsupportedImageError):
            vision.prepare_image(b"\x00\x01garbage")


# ─── extraction ───────────────────────────────────────────────────────────────

def _fake_vision(kind_log: list):
    def ocr(data, label=""):
        kind_log.append(("ocr", label))
        return vision.VisionResult(text=ARABIC + "\n[وصف صورة] رسم توضيحي")

    def describe(data, label=""):
        kind_log.append(("describe", label))
        return vision.VisionResult(text="[وصف صورة] مخطط لدورة الماء مع أسهم")

    return ocr, describe


class TestExtractPages:
    def test_text_pdf_needs_no_vision(self):
        calls: list = []
        ocr, describe = _fake_vision(calls)
        with patch.object(vision, "ocr_image", ocr), patch.object(vision, "describe_image", describe):
            result = extractor.extract_pages(_text_pdf([ENGLISH_PAGE, ENGLISH_PAGE[:10]]), "book.pdf")
        assert result.page_count == 2 and result.vision_pages == 0 and not result.truncated
        assert [p.method for p in result.pages] == ["text", "text"]
        assert "photosynthesis" in result.pages[0].text
        assert calls == []
        assert result.method == "text"

    def test_scanned_page_is_ocred_and_figures_described(self):
        calls: list = []
        ocr, describe = _fake_vision(calls)
        pdf = _merge(_text_pdf([ENGLISH_PAGE]), _image_pdf(), _overlay(_text_pdf([ENGLISH_PAGE]), _image_pdf()))
        with patch.object(vision, "ocr_image", ocr), patch.object(vision, "describe_image", describe):
            result = extractor.extract_pages(pdf, "book.pdf")
        methods = [p.method for p in result.pages]
        assert methods == ["text", "ocr", "text+vision"], methods
        assert ARABIC in result.pages[1].text
        assert result.pages[2].text.endswith("[وصف صورة] مخطط لدورة الماء مع أسهم")
        assert result.vision_pages == 2
        assert sorted(kind for kind, _ in calls) == ["describe", "ocr"]
        assert result.method == "mixed"

    def test_budget_caps_vision(self):
        calls: list = []
        ocr, describe = _fake_vision(calls)
        pdf = _merge(_image_pdf(), _image_pdf())
        with patch.object(vision, "ocr_image", ocr), patch.object(vision, "describe_image", describe):
            result = extractor.extract_pages(pdf, "scan.pdf", max_vision_pages=1)
        assert result.truncated and result.vision_pages == 1 and len(calls) == 1

    def test_vision_disabled_for_unflagged_use(self):
        with patch.object(vision, "ocr_image", side_effect=AssertionError("no vision")):
            result = extractor.extract_pages(_merge(_text_pdf([ENGLISH_PAGE]), _image_pdf()), "b.pdf",
                                             vision_enabled=False)
        assert result.pages[1].text == "" and result.vision_pages == 0

    def test_all_ocr_failing_raises(self):
        with patch.object(vision, "ocr_image", side_effect=vision.VisionError("down")):
            with pytest.raises(vision.VisionError):
                extractor.extract_pages(_image_pdf(), "scan.pdf")

    def test_image_file(self):
        with patch.object(vision, "ocr_image", return_value=vision.VisionResult(text="ـنص  الصورة")):
            result = extractor.extract_pages(_png(), "photo.png")
        assert result.pages[0].text == "نص الصورة" and result.pages[0].method == "ocr"

    def test_unsupported_type(self):
        with pytest.raises(ValueError):
            extractor.extract_pages(b"PK\x03\x04", "archive.zip")


class TestLegacyImageOcr:
    def test_extract_text_image_goes_to_deepseek(self):
        with patch.object(vision, "ocr_image", return_value=vision.VisionResult(text="hello")) as ocr:
            assert extractor.extract_text(_png(fmt="GIF"), "pic.gif") == "hello"
        ocr.assert_called_once()


# ─── POST /extract ────────────────────────────────────────────────────────────

class TestExtractEndpoint:
    def test_base64_pdf(self):
        pdf = _text_pdf([ENGLISH_PAGE])
        resp = client.post("/extract", headers=DAFTAR, json={
            "fileName": "a.pdf", "fileBase64": base64.b64encode(pdf).decode()})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["pageCount"] == 1 and body["visionPages"] == 0 and body["truncated"] is False
        assert body["pages"][0]["method"] == "text" and "photosynthesis" in body["pages"][0]["text"]

    def test_text_file(self):
        resp = client.post("/extract", headers=DAFTAR, json={
            "fileName": "notes.txt", "fileBase64": base64.b64encode("سطر أول\nسطر ثان".encode()).decode()})
        assert resp.status_code == 200
        assert resp.json()["pages"] == [{"page": 1, "text": "سطر أول\nسطر ثان", "method": "text"}]

    @pytest.mark.parametrize("body", [
        {"fileName": "a.pdf"},
        {"fileName": "a.pdf", "fileBase64": "QQ==", "storageUrl": "https://x"},
        {"fileName": "a.pdf", "fileBase64": "not base64!!"},
    ])
    def test_bad_requests(self, body):
        assert client.post("/extract", headers=DAFTAR, json=body).status_code == 422

    def test_too_large(self):
        big = base64.b64encode(b"\0" * (15 * 1024 * 1024 + 10)).decode()
        assert client.post("/extract", headers=DAFTAR, json={"fileName": "a.pdf", "fileBase64": big}).status_code == 422

    def test_needs_key(self):
        assert client.post("/extract", json={"fileName": "a.txt", "fileBase64": "QQ=="}).status_code == 401

    def test_vision_failure_is_502(self):
        with patch.object(vision, "ocr_image", side_effect=vision.VisionError("down")):
            resp = client.post("/extract", headers=DAFTAR, json={
                "fileName": "scan.pdf", "fileBase64": base64.b64encode(_image_pdf()).decode()})
        assert resp.status_code == 502
