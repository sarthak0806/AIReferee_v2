"""PDF text extraction with section/page structure.

Primary: PyMuPDF (font-aware heading detection, block order for two-column
layouts, header/footer removal). Fallback: pypdf (regex-only headings).
Scanned PDFs without a text layer are reported as ``needs_ocr``; OCR is not
bundled (it needs a Tesseract install).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
from pypdf import PdfReader

from config import CONFIG, path
from ingestion.section_parser import canonical_section, is_acknowledgement, parse_heading
from utils.logging_config import get_logger
from utils.text_utils import normalize_text

log = get_logger("extractor")
# pypdf logs recoverable structure problems ("Ignoring wrong pointing object") as warnings.
import logging as _logging
_logging.getLogger("pypdf").setLevel(_logging.ERROR)

BOLD = 1 << 4

# Publisher/venue boilerplate. Removed because it is not research content and
# would leak the venue label into retrieval and conference matching.
_BOILERPLATE = [re.compile(p, re.I | re.S) for p in [
    # CVF watermark (often split across two text blocks, so each sentence separately)
    r"This (CVPR|ICCV|WACV|ECCV)[^.]{0,40}paper is the Open Access version,? provided by the Computer Vision Foundation\.?",
    r"Except for this watermark,? it is identical to the accepted version.{0,200}?IEEE Xplore\.?",
    # ACM / KDD
    r"Permission to make digital or hard copies.{0,1200}?(permissions@acm\.org\.?|$)",
    r"©\s?20\d\d\s+(Copyright held by the owner/author\(s\)|Association for Computing Machinery)[^.]*\.?",
    r"(Publication rights licensed to ACM|Copyright held by the owner/author\(s\))\.?",
    r"ACM ISBN [\d\-/ .]+",
    r"https?://doi\.org/10\.1145/[\d.]+",
    r"ACM Reference Format:.{0,900}?(20\d\d,\s*[A-Z][\w .]{1,40},\s*[A-Z][\w ]{1,30}\.|\d+ pages\.|$)",
    r"ACM,\s*(New York, NY, USA|[A-Z][\w ]{1,30}, [A-Z]{2,4}, [A-Z][\w ]{1,20}),\s*\d+\s*pages\.?",
    r"In Proceedings of the \d+(st|nd|rd|th) ACM SIGKDD Conference.{0,200}?20\d\d,[^.]{0,60}\.",
    r"\bKDD\s?[’'`]\s?\d\d,\s+\w+ \d+[–\-]?\d*,\s+20\d\d,[^.\n]{0,60}",
    r"CCS CONCEPTS\s*•?",
    # NeurIPS / EMNLP / TMLR footers and headers
    r"\d{1,2}(st|nd|rd|th) Conference on Neural Information Processing Systems \(NeurIPS 20\d\d\)[^.]*\.?",
    r"Proceedings of the 20\d\d Conference on Empirical Methods in Natural Language Processing,? pages [\d–\-]+.{0,120}?Linguistics\.?",
    r"Published in Transactions on Machine Learning Research \(\d\d/20\d\d\)",
    r"Reviewed on OpenReview:\s*https?\s*:\s*//\s*openreview\.\s*net/\s*forum\?\s*id=\s*\S+",
    r"Under review as (a )?submission to TMLR",
]]


def strip_boilerplate(text: str) -> str:
    for pat in _BOILERPLATE:
        text = pat.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


@dataclass
class Paragraph:
    page: int
    text: str


@dataclass
class Section:
    name: str                # canonical, e.g. "Methodology"
    heading: str             # raw heading as printed
    page_start: int
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)


@dataclass
class ExtractedDocument:
    title: str
    abstract: str
    sections: list[Section]
    page_count: int
    method: str              # pymupdf | pypdf
    char_count: int

    def section_text(self, name: str) -> str:
        return "\n".join(s.text for s in self.sections if s.name == name)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "ExtractedDocument":
        secs = [Section(s["name"], s["heading"], s["page_start"], [Paragraph(**p) for p in s["paragraphs"]])
                for s in d["sections"]]
        return cls(d["title"], d["abstract"], secs, d["page_count"], d["method"], d["char_count"])


class ExtractionError(RuntimeError):
    def __init__(self, status: str, msg: str) -> None:
        super().__init__(msg)
        self.status = status


# ---------------------------------------------------------------------------
def _join_spans(spans: list[dict]) -> str:
    """Join a line's spans. Whitespace-only spans are real word separators, and a
    horizontal gap between spans also means a word boundary (no space glyph)."""
    out, prev_x1 = "", None
    for s in spans:
        t = s["text"]
        if prev_x1 is not None and t and out and not out.endswith(" ") and not t.startswith(" "):
            gap = s["bbox"][0] - prev_x1
            if gap > 0.15 * s["size"]:
                out += " "
        out += t
        prev_x1 = s["bbox"][2]
    return re.sub(r"\s+", " ", out).strip()


def _page_lines_pymupdf(doc: pymupdf.Document) -> list[list[tuple[str, bool, float, bool]]]:
    """Per page: (line_text, emphasised, font_size, block_start)."""
    sizes: Counter = Counter()
    pages = []
    for page in doc:
        lines = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            first = True
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = _join_spans(line["spans"])
                size = max(s["size"] for s in spans)
                bold = all(s["flags"] & BOLD or "bold" in s["font"].lower() or "medi" in s["font"].lower()
                           for s in spans)
                sizes[round(size, 1)] += len(text)
                lines.append((text, bold, size, first))
                first = False
        pages.append(lines)
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    out = []
    for lines in pages:
        lines = [(t, b or s > body + 0.8, s, f) for (t, b, s, f) in lines]
        merged: list[tuple[str, bool, float, bool]] = []
        i = 0
        while i < len(lines):
            t, e, s, f = lines[i]
            # "2" / "2.1" printed on its own line followed by the heading title
            if e and _BARE_NUMBER.match(t) and i + 1 < len(lines) and lines[i + 1][1]:
                nt = lines[i + 1][0]
                merged.append((f"{t} {nt}", True, max(s, lines[i + 1][2]), f))
                i += 2
                continue
            merged.append((t, e, s, f))
            i += 1
        out.append(merged)
    return out


_BARE_NUMBER = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,3}|[A-H](\.\d{1,2})*|[IVX]{1,5})\.?$")


def _page_lines_pypdf(pdf: Path) -> list[list[tuple[str, bool, float, bool]]]:
    reader = PdfReader(str(pdf))
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append([(ln.strip(), False, 0.0, True) for ln in txt.splitlines() if ln.strip()])
    return pages


def _repeated_lines(pages: list[list[tuple]]) -> set[str]:
    """Running headers/footers: identical short lines on > 40% of pages."""
    if len(pages) < 3:
        return set()
    counts = Counter()
    for lines in pages:
        counts.update({re.sub(r"\d+", "#", t) for t, *_ in lines if len(t) < 120})
    return {t for t, n in counts.items() if n > 0.4 * len(pages)}


def _infer_section(seen: set[str], heading: str) -> str:
    """Map a paper-specific top-level heading (e.g. '3 Sparse Mechanism Shift VAE')
    to a canonical section from its position in the paper."""
    if "References" in seen:
        return "Appendix"
    if "Conclusion" in seen:
        return "Discussion"
    if seen & {"Experiments", "Results"}:
        return "Results"
    if "Introduction" in seen:
        return "Methodology"
    return re.sub(r"^[\dIVXA-H.]+\s+", "", heading).strip()


def _build_sections(pages: list[list[tuple[str, bool, float, bool]]]) -> tuple[list[Section], str]:
    repeated = _repeated_lines(pages)
    sections: list[Section] = [Section("Front Matter", "", 1)]
    current_top = "Front Matter"
    buf: list[str] = []
    buf_page = 1
    title_candidates: list[tuple[float, str]] = []
    seen: set[str] = set()

    def flush():
        nonlocal buf
        if buf:
            text = strip_boilerplate(normalize_text("\n".join(buf)))
            if len(text) > 1:
                sections[-1].paragraphs.append(Paragraph(buf_page, text))
        buf = []

    for pno, lines in enumerate(pages, start=1):
        for text, emph, size, block_start in lines:
            if re.sub(r"\d+", "#", text) in repeated or re.fullmatch(r"[\d\s]+", text):
                continue
            if pno == 1 and len(sections) == 1 and size:
                title_candidates.append((size, text))
            head = parse_heading(text, emph)
            if head:
                heading, level = head
                canon = canonical_section(heading)
                if is_acknowledgement(heading):
                    canon = "Acknowledgements"
                # Before the abstract/introduction, numbered bold lines are affiliations.
                if current_top == "Front Matter" and canon not in ("Abstract", "Introduction"):
                    head = None
                elif not re.search(r"[A-Za-z]{3}", heading):
                    head = None
            if head:
                if level == 1 or (canon in ("Abstract", "References") and level <= 2):
                    current_top = canon or _infer_section(seen, heading)
                    seen.add(current_top)
                    name = current_top
                else:
                    name = current_top  # subsections inherit their parent's canonical section
                flush()
                sections.append(Section(name, heading, pno))
                buf_page = pno
                continue
            # "Abstract—Text ..." / "Abstract. Text" run-in style
            m = re.match(r"^(abstract)\s*[—:.\-–]\s*(.+)$", text, re.I)
            if m and current_top == "Front Matter":
                flush()
                current_top = "Abstract"
                sections.append(Section("Abstract", "Abstract", pno))
                text = m.group(2)
            if block_start:
                flush()
                buf_page = pno
            buf.append(text)
        flush()
        buf_page = pno + 1

    title = ""
    if title_candidates:
        top = max(s for s, _ in title_candidates)
        title = " ".join(t for s, t in title_candidates if s == top)[:300]
    return [s for s in sections if s.paragraphs], title


def extract_pdf(pdf_path: str | Path) -> ExtractedDocument:
    pdf_path = Path(pdf_path)
    method = "pymupdf"
    try:
        with pymupdf.open(pdf_path) as doc:
            page_count = doc.page_count
            pages = _page_lines_pymupdf(doc)
    except Exception as exc:
        log.warning("PyMuPDF failed on %s (%s); trying pypdf", pdf_path.name, exc)
        try:
            pages = _page_lines_pypdf(pdf_path)
        except Exception as exc2:
            raise ExtractionError("failed", f"unreadable PDF: {exc2}") from exc2
        page_count, method = len(pages), "pypdf"

    chars = sum(len(t) for lines in pages for t, *_ in lines)
    if chars < CONFIG["min_extractable_chars"] and method == "pymupdf":
        pages = _page_lines_pypdf(pdf_path)
        method = "pypdf"
        chars = sum(len(t) for lines in pages for t, *_ in lines)
    if chars < CONFIG["min_extractable_chars"]:
        raise ExtractionError("needs_ocr", f"only {chars} characters of text; scanned PDF needs OCR")

    sections, title = _build_sections(pages)
    named = {s.name for s in sections}
    if len(named - {"Front Matter"}) < 2:
        log.warning("%s: few headings detected (%s); text kept as generic sections", pdf_path.name, named)

    abstract = "\n".join(s.text for s in sections if s.name == "Abstract")
    if not abstract:  # fall back to the first substantial front-matter paragraph
        front = [p.text for s in sections if s.name == "Front Matter" for p in s.paragraphs if len(p.text) > 400]
        abstract = front[0] if front else ""
    total = sum(len(s.text) for s in sections)
    if total < CONFIG["min_extractable_chars"]:
        raise ExtractionError("failed", f"extraction produced only {total} characters")
    return ExtractedDocument(title, abstract, sections, page_count, method, total)


def cache_path(paper_id: str) -> Path:
    return path("extracted_dir") / "papers" / f"{paper_id}.json"


def save_extraction(paper_id: str, doc: ExtractedDocument) -> Path:
    p = cache_path(paper_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc.to_json(), ensure_ascii=False), encoding="utf-8")
    return p


def load_extraction(paper_id: str) -> ExtractedDocument | None:
    p = cache_path(paper_id)
    if not p.exists():
        return None
    return ExtractedDocument.from_json(json.loads(p.read_text(encoding="utf-8")))
