"""
Unit tests for the page-aware chunker (pages_v1), the text-quality grader and
log redaction. No network, no Chroma.
"""

from app.log_utils import redact_body, redact_url
from app.services.chunker import (
    PAGES_MAX_CHARS,
    PAGES_MIN_CHARS,
    PAGES_OVERLAP_CHARS,
    PAGES_TARGET_CHARS,
    chunk_pages,
)
from app.services.text_quality import assess_text, clean_text, normalize_light


def _words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


# ─── chunk_pages ──────────────────────────────────────────────────────────────

class TestChunkPages:
    def test_metadata_and_indexes(self):
        pages = [(1, _words(400, "a")), (2, _words(400, "b")), (3, _words(50, "c"))]
        chunks = chunk_pages(pages, "book.pdf", "nb1", "src1")
        assert chunks
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            assert set(meta) == {"notebookId", "sourceId", "source", "chunkIndex", "pageStart", "pageEnd"}
            assert meta["notebookId"] == "nb1" and meta["sourceId"] == "src1" and meta["source"] == "book.pdf"
            assert meta["chunkIndex"] == i
            assert 1 <= meta["pageStart"] <= meta["pageEnd"] <= 3
        assert chunks[0]["metadata"]["pageStart"] == 1
        assert chunks[-1]["metadata"]["pageEnd"] == 3

    def test_sizes_within_bounds(self):
        pages = [(p, _words(300 + 37 * p, f"p{p}w")) for p in range(1, 15)]
        chunks = chunk_pages(pages, "f.pdf", "nb", "s")
        assert len(chunks) > 3
        for chunk in chunks:
            assert len(chunk["content"]) <= PAGES_MAX_CHARS
        # Every chunk but possibly the last carries at least the minimum of its own text.
        for chunk in chunks[:-1]:
            assert len(chunk["content"]) >= PAGES_MIN_CHARS

    def test_long_page_split_and_page_span(self):
        text = "\n".join(_words(20, f"line{i}x") for i in range(60))  # ~one long page
        chunks = chunk_pages([(7, text)], "f.pdf", "nb", "s")
        assert len(chunks) >= 2
        assert all(c["metadata"]["pageStart"] == c["metadata"]["pageEnd"] == 7 for c in chunks)
        for chunk in chunks:
            assert len(chunk["content"]) <= PAGES_MAX_CHARS

    def test_short_pages_packed(self):
        pages = [(p, f"Short page {p} " + _words(20, "w")) for p in range(1, 11)]
        chunks = chunk_pages(pages, "f.pdf", "nb", "s")
        # ten ~150-char pages fit in one or two target-sized chunks
        assert len(chunks) <= 2
        assert chunks[0]["metadata"]["pageStart"] == 1
        assert chunks[-1]["metadata"]["pageEnd"] == 10

    def test_overlap_prefix_from_previous_chunk(self):
        pages = [(1, _words(600, "a"))]
        chunks = chunk_pages(pages, "f.pdf", "nb", "s")
        assert len(chunks) >= 2
        for prev, cur in zip(chunks, chunks[1:]):
            head = cur["content"][:PAGES_OVERLAP_CHARS]
            first_word = head.split()[0]
            # the overlap starts on a whole word that ends the previous chunk's text
            assert first_word in prev["content"].split()
        # all source words survive, in order
        joined = " ".join(c["content"] for c in chunks)
        assert "a0" in joined and "a599" in joined

    def test_empty_pages_skipped(self):
        assert chunk_pages([(1, "   "), (2, "")], "f.pdf", "nb", "s") == []
        chunks = chunk_pages([(1, ""), (2, _words(50, "x"))], "f.pdf", "nb", "s")
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["pageStart"] == 2

    def test_target_size_is_respected_roughly(self):
        chunks = chunk_pages([(1, _words(3000, "w"))], "f.pdf", "nb", "s")
        bases = [len(c["content"]) for c in chunks]
        # own text near the target, plus at most the overlap
        assert max(bases) <= PAGES_TARGET_CHARS + PAGES_OVERLAP_CHARS + 2

    def test_arabic_text(self):
        para = "الوحدة الأولى: الأعداد الحقيقية. " * 80
        chunks = chunk_pages([(1, para), (2, para)], "كتاب.pdf", "nb", "s")
        assert len(chunks) >= 2
        assert all("الأعداد" in c["content"] for c in chunks)


# ─── text quality ─────────────────────────────────────────────────────────────

ARABIC_OK = (
    "الدرس الأول: مفهوم العدد النسبي\n"
    "العدد النسبي هو كل عدد يمكن كتابته على صورة كسر بسطه ومقامه عددان صحيحان "
    "والمقام لا يساوي صفراً. أمثلة على الأعداد النسبية كثيرة في حياتنا اليومية."
)


class TestTextQuality:
    def test_ok_arabic(self):
        q = assess_text(ARABIC_OK)
        assert q.status == "ok", q.reasons
        assert q.usable

    def test_glyph_names_are_junk(self):
        q = assess_text("/g875/g257/g3/g1022 /g44/g90/g12 " * 30)
        assert q.status == "junk"
        assert "glyph-names" in q.reasons
        assert not q.usable

    def test_latin1_mojibake_is_junk(self):
        q = assess_text("ÚÊ≈Ë ¨dO… ÚÊ«ÿ ˆÒ Ê ÷ÕÂ ÆÚ «Ëœ ÌÒ” " * 20)
        assert q.status == "junk"
        assert "mojibake" in q.reasons

    def test_control_chars_are_junk(self):
        q = assess_text("\x01\x02\x03abc\x04\x05\x06\x07 " * 40)
        assert q.status == "junk"

    def test_replacement_chars(self):
        assert assess_text("�" * 50 + " abc").status == "junk"
        text = ARABIC_OK + " " + "كلمة�" * 3
        q = assess_text(text)
        assert q.status in ("degraded", "ok")
        assert "�" not in q.text

    def test_empty(self):
        assert assess_text("").status == "empty"
        assert assess_text("  12 \n ").status == "empty"

    def test_broken_words_degraded(self):
        q = assess_text(" ".join("ا ل م د ر س ة" for _ in range(20)))
        assert q.status == "degraded"
        assert "broken-words" in q.reasons

    def test_clean_text_folds_presentation_forms_and_tatweel(self):
        # U+FEDF U+FEE0 (lam forms) and a tatweel
        cleaned = clean_text("ﻟﻠـــعـــلـــم​ x²")
        assert "ـ" not in cleaned
        assert "​" not in cleaned
        assert "x²" in cleaned  # NFKC applied only to presentation forms
        assert "ﻟ" not in cleaned

    def test_normalize_light_keeps_newlines(self):
        text = "Title\r\n\r\n\r\n\r\nline one   with   spaces\x00\nline\ttwo‏"
        out = normalize_light(text)
        assert out == "Title\n\nline one with spaces\nline two"


# ─── log redaction ────────────────────────────────────────────────────────────

class TestRedaction:
    URL = (
        "https://firebasestorage.googleapis.com/v0/b/bucket/o/notebooks%2Fa.pdf"
        "?alt=media&token=1234-secret-5678"
    )

    def test_redact_url(self):
        out = redact_url(self.URL)
        assert "1234-secret-5678" not in out
        assert "token=<redacted>" in out
        assert "alt=media" in out

    def test_redact_body_storage_url(self):
        body = '{"storageUrl": "%s", "fileName": "a.pdf"}' % self.URL
        out = redact_body("/ingest", body)
        assert "1234-secret-5678" not in out
        assert "a.pdf" in out

    def test_redact_body_base64(self):
        body = '{"fileName": "a.png", "fileBase64": "%s"}' % ("QUJD" * 5000)
        out = redact_body("/extract", body)
        assert "QUJDQUJD" not in out
        assert "20000 base64 chars" in out

    def test_redact_body_ingest_text_is_summarised(self):
        import json

        body = json.dumps({
            "notebookId": "nb", "sourceId": "s", "fileName": "f.pdf",
            "pages": [{"page": 1, "text": "secret page text " * 100}],
        })
        out = redact_body("/ingest/text", body)
        assert "secret page text" not in out
        data = json.loads(out)
        assert data["pages"] == 1 and data["chars"] == len("secret page text " * 100)

    def test_long_body_truncated(self):
        out = redact_body("/retrieve", "x" * 10_000)
        assert len(out) < 2_100
        assert "truncated" in out
