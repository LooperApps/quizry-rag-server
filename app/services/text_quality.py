"""
Text normalisation and text-layer quality checks for page-wise extraction.

Many school-book PDFs embed fonts with no usable ToUnicode map, so their text
layer extracts as garbage that still *looks* like text: glyph names
(`/g875/g257`), Latin-1 mojibake of Arabic (`ÚÊ≈Ë ¨dO…`), control characters,
U+FFFD replacement characters, or Quran-font glyphs that decode to strings of
isolated letters. Indexing that text pollutes retrieval, so pages are graded:

  ok        usable as-is
  degraded  readable but damaged (a few % U+FFFD, broken words) — OCR it when
            the vision budget allows, otherwise keep the cleaned text
  junk      unreadable — OCR it, or drop it when the budget is spent
  empty     (almost) no text — OCR it when the page shows something

Only the page-wise paths (flagged collections, POST /extract) use this module;
legacy collections index exactly what pypdf returns.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Arabic/Latin/Hebrew presentation forms (ligatures, positional letter forms).
# NFKC folds them to the base letters; applying it only here leaves math such
# as "x²" alone, which a whole-text NFKC would flatten to "x2".
_PRESENTATION_FORMS = re.compile("[ﬀ-﷿ﹰ-﻾]+")
_TATWEEL = "ـ"
# C0/C1 controls except tab and newline.
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# Invisible format characters: soft hyphen, zero-width space, LRM/RLM, bidi
# embeddings/isolates, word joiner, BOM. ZWNJ/ZWJ are kept (meaningful in
# some scripts).
_INVISIBLE = re.compile("[­​‎‏‪-‮⁠⁦-⁩﻿]")
_SPACES = re.compile(r"[ \t   -   　]+")
_BLANK_LINES = re.compile(r"\n{3,}")

# pypdf renders unmapped glyphs as their names: /g875, /I103+, /uni0627 or
# control-code names such as /DC3.
_GLYPH_TOKEN = re.compile(
    r"/(?:[A-Za-z]{1,2}[0-9+]{1,6}|uni[0-9A-Fa-f]{4,6}"
    r"|(?:NUL|SOH|STX|ETX|EOT|ENQ|ACK|BEL|BS|VT|FF|SO|SI|DLE|NAK|SYN|ETB|CAN|EM|SUB|ESC|FS|GS|RS|US|DEL)(?![A-Za-z]))"
)
# Latin-1 / cp1252 characters that Arabic legacy-font mojibake is made of.
# Excludes symbols real text uses (« » ° ± ² ³ µ · ¼ ½ ¾ × ÷ © ® §).
_MOJIBAKE_EXTRA = set("ŒœŠšŸŽžƒˆ˜‰‹›†‡")
_LATIN1_OK = set("«»°±²³µ·¼½¾×÷©®§ ")

# Thresholds, as shares of the non-space characters of a page.
MIN_TEXT_CHARS = 20          # fewer letters/digits than this: the page is "empty"
GLYPH_JUNK = 0.15
CONTROL_JUNK = 0.05
PUA_JUNK = 0.10
MOJIBAKE_JUNK = 0.08         # ... while Arabic letters stay under ARABIC_FLOOR
ARABIC_FLOOR = 0.30
FFFD_JUNK = 0.15
FFFD_DEGRADED = 0.02
LETTERS_JUNK = 0.30          # letters+digits below this share
SINGLE_LETTER_DEGRADED = 0.5  # share of 1-letter words, with >= 30 words


def normalize_light(text: str) -> str:
    """
    Light clean-up for text that is already good (POST /ingest/text): drop
    control and invisible characters, collapse runs of spaces, trim lines and
    blank-line runs. Newlines are kept; nothing else is rewritten.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub(" ", text)
    text = _INVISIBLE.sub("", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def clean_text(text: str) -> str:
    """
    Full clean-up for extracted page text: presentation forms folded with
    NFKC, tatweel and U+FFFD removed, then normalize_light.
    """
    text = _PRESENTATION_FORMS.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    text = text.replace(_TATWEEL, "").replace("�", "")
    return normalize_light(text)


def _is_arabic(o: int) -> bool:
    return (0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF
            or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFC)


@dataclass
class TextQuality:
    status: str                       # ok | degraded | junk | empty
    text: str                         # cleaned text (clean_text)
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Text worth indexing when no OCR replaces it."""
        return self.status in ("ok", "degraded")


def assess_text(raw: str) -> TextQuality:
    """Grade one page's extracted text layer and return it cleaned."""
    cleaned = clean_text(raw)
    chars = [ch for ch in raw if not ch.isspace()]
    n = len(chars)
    arabic = control = pua = fffd = mojibake = letters = 0
    for ch in chars:
        o = ord(ch)
        if _is_arabic(o):
            arabic += 1
        if o < 32 or 0x7F <= o <= 0x9F:
            control += 1
        elif 0xE000 <= o <= 0xF8FF:
            pua += 1
        elif o == 0xFFFD:
            fffd += 1
        elif (0xA1 <= o <= 0xFF and ch not in _LATIN1_OK) or ch in _MOJIBAKE_EXTRA:
            mojibake += 1
        elif ch.isalnum():
            letters += 1
    glyph = sum(len(m.group()) for m in _GLYPH_TOKEN.finditer(raw))

    def share(count: int) -> float:
        return round(count / n, 4) if n else 0.0

    stats = {
        "chars": n,
        "arabic": share(arabic),
        "letters": share(letters),
        "glyph": share(glyph),
        "control": share(control),
        "pua": share(pua),
        "fffd": share(fffd),
        "mojibake": share(mojibake),
    }

    reasons: list[str] = []
    if stats["glyph"] >= GLYPH_JUNK:
        reasons.append("glyph-names")
    if stats["control"] >= CONTROL_JUNK:
        reasons.append("control-chars")
    if stats["pua"] >= PUA_JUNK:
        reasons.append("private-use")
    if stats["mojibake"] >= MOJIBAKE_JUNK and stats["arabic"] < ARABIC_FLOOR:
        reasons.append("mojibake")
    if stats["fffd"] >= FFFD_JUNK:
        reasons.append("replacement-chars")
    if n and not reasons and stats["letters"] < LETTERS_JUNK:
        reasons.append("few-letters")
    if reasons:
        return TextQuality("junk", cleaned, reasons, stats)

    if sum(1 for ch in cleaned if ch.isalnum()) < MIN_TEXT_CHARS:
        return TextQuality("empty", cleaned, ["no-text"], stats)

    words = [w for w in cleaned.split() if any(ch.isalpha() for ch in w)]
    single = sum(1 for w in words if sum(1 for ch in w if ch.isalpha()) == 1)
    stats["singleLetterWords"] = round(single / len(words), 4) if words else 0.0
    if stats["fffd"] >= FFFD_DEGRADED:
        reasons.append("replacement-chars")
    if len(words) >= 30 and stats["singleLetterWords"] >= SINGLE_LETTER_DEGRADED:
        reasons.append("broken-words")
    if reasons:
        return TextQuality("degraded", cleaned, reasons, stats)
    return TextQuality("ok", cleaned, [], stats)
