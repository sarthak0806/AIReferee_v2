"""PDF download and integrity validation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

from config import CONFIG
from utils.http import client
from utils.logging_config import get_logger

log = get_logger("downloader")


@dataclass
class PdfCheck:
    ok: bool
    status: str          # valid | invalid | needs_ocr
    detail: str
    page_count: int = 0
    sha256: str | None = None


def validate_pdf(data: bytes) -> PdfCheck:
    """A PDF is valid when it opens, has enough pages and has extractable text."""
    sha = hashlib.sha256(data).hexdigest()
    if not data.startswith(b"%PDF"):
        return PdfCheck(False, "invalid", "not a PDF (missing %PDF header)", sha256=sha)
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.is_encrypted and not doc.authenticate(""):
                return PdfCheck(False, "invalid", "encrypted PDF", sha256=sha)
            pages = doc.page_count
            if pages < CONFIG["min_pdf_pages"]:
                return PdfCheck(False, "invalid", f"only {pages} pages", pages, sha)
            text = "".join(doc[i].get_text() for i in range(min(3, pages)))
    except Exception as exc:  # corrupted file
        return PdfCheck(False, "invalid", f"cannot open PDF: {exc}", sha256=sha)
    chars = len(text.strip())
    if chars < CONFIG["min_extractable_chars"]:
        return PdfCheck(False, "needs_ocr", f"only {chars} extractable chars on first pages", pages, sha)
    return PdfCheck(True, "valid", f"{pages} pages, {chars} chars on first pages", pages, sha)


def download_pdf(url: str, dest: Path) -> tuple[PdfCheck | None, str]:
    """Download (or reuse an existing file) and validate. Returns (check, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        data = dest.read_bytes()
        return validate_pdf(data), "reused existing file"
    data = client().download(url, CONFIG["max_pdf_mb"] * 1024 * 1024)
    if data is None:
        return None, f"download failed: {url}"
    check = validate_pdf(data)
    if check.ok:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
    return check, "downloaded"
