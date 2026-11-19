"""Text helpers shared across ingestion, retrieval and reporting."""
from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
# Abbreviations that should not end a sentence.
_ABBREV = re.compile(r"\b(e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Sec|Tab|vs|cf|resp|approx|No|Ref|Refs)\.$", re.I)


def normalize_text(text: str) -> str:
    """NFKC-normalise, drop control chars, de-hyphenate line breaks, collapse spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    return re.sub(r"[ \t ]+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def word_count(text: str) -> int:
    return len((text or "").split())


def truncate_words(text: str, max_words: int) -> str:
    """Trim to at most ``max_words`` words, preferring a sentence boundary."""
    words = (text or "").split()
    if len(words) <= max_words:
        return (text or "").strip()
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind(".") if cut.endswith(".") else -1)
    if end > len(cut) * 0.6:
        return cut[: end + 1].strip()
    return cut.rstrip(",;: ") + "."
