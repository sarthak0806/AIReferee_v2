import pymupdf
import pytest

from ingestion.extractor import ExtractionError, extract_pdf, strip_boilerplate
from ingestion.section_parser import canonical_section, parse_heading

BODY = ("This sentence describes the research in reasonable detail so that extraction has text to work with. " * 6)


def make_pdf(tmp_path, sections, bold=True, number_on_own_line=False):
    doc = pymupdf.open()
    font = "hebo" if bold else "helv"
    page = doc.new_page()
    page.insert_text((72, 60), "A Study of Robust Things", fontsize=16, fontname="hebo")
    y = 100
    for i, (heading, text) in enumerate(sections, 1):
        if y > 650:
            page = doc.new_page()
            y = 72
        if number_on_own_line:
            page.insert_text((72, y), str(i), fontsize=11, fontname=font)
            y += 14
            page.insert_text((72, y), heading, fontsize=11, fontname=font)
        else:
            page.insert_text((72, y), f"{i} {heading}" if heading != "Abstract" else heading, fontsize=11, fontname=font)
        y += 18
        rect = pymupdf.Rect(72, y, 520, y + 160)
        page.insert_textbox(rect, text, fontsize=9, fontname="helv")
        y += 175
    p = tmp_path / "paper.pdf"
    doc.save(p)
    return p


@pytest.mark.parametrize("line,emph,expected", [
    ("1 Introduction", True, ("1 Introduction", 1)),
    ("3.2 Training Details", True, ("3.2 Training Details", 2)),
    ("III. EXPERIMENTS", True, ("III. EXPERIMENTS", 1)),
    ("References", False, ("References", 1)),
    ("A Simple Baseline for Everything", True, None),       # title-like, not a numbered heading
    ("1 Introduction", False, None),                         # numbered but not emphasised
    ("This is a long sentence that is clearly body text and not a heading at all, ok.", True, None),
])
def test_parse_heading(line, emph, expected):
    assert parse_heading(line, emph) == expected


def test_camel_case_heading_restored():
    assert parse_heading("2 RelatedWork", True)[0] == "2 Related Work"


@pytest.mark.parametrize("heading,canon", [
    ("4 Experimental Setup", "Experiments"), ("5 Results and Analysis", "Results"),
    ("2 Related Work", "Related Work"), ("3 Proposed Method", "Methodology"),
    ("6 Conclusions and Future Work", "Conclusion"), ("References", "References"),
    ("Limitations", "Limitations"), ("7 Discussion", "Discussion"),
])
def test_canonical_section(heading, canon):
    assert canonical_section(heading) == canon


def test_strip_boilerplate():
    t = ("This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this "
         "watermark, it is identical to the accepted version; the final published version of the proceedings is "
         "available on IEEE Xplore. Real content stays.")
    assert strip_boilerplate(t) == "Real content stays."
    acm = "Permission to make digital or hard copies of all or part of this work ... Request permissions from permissions@acm.org. Body."
    assert strip_boilerplate(acm) == "Body."


def test_extract_sections_and_pages(tmp_path):
    secs = [("Abstract", BODY), ("Introduction", BODY), ("Method", BODY), ("Experiments", BODY),
            ("Conclusion", BODY), ("References", "[1] A. Author. A paper. 2020.")]
    doc = extract_pdf(make_pdf(tmp_path, secs))
    names = [s.name for s in doc.sections]
    for expected in ("Abstract", "Introduction", "Methodology", "Experiments", "Conclusion", "References"):
        assert expected in names
    assert doc.abstract.startswith("This sentence")
    assert all(p.page >= 1 for s in doc.sections for p in s.paragraphs)
    assert doc.page_count >= 2


def test_number_on_separate_line_is_merged(tmp_path):
    secs = [("Introduction", BODY), ("Related Work", BODY), ("Method", BODY), ("Experiments", BODY)]
    doc = extract_pdf(make_pdf(tmp_path, secs, number_on_own_line=True))
    assert {"Introduction", "Related Work", "Experiments"} <= {s.name for s in doc.sections}


def test_empty_pdf_needs_ocr(tmp_path):
    d = pymupdf.open()
    for _ in range(5):
        d.new_page()
    p = tmp_path / "blank.pdf"
    d.save(p)
    with pytest.raises(ExtractionError) as e:
        extract_pdf(p)
    assert e.value.status == "needs_ocr"


def test_join_spans_inserts_spaces_at_gaps():
    from ingestion.extractor import _join_spans
    spans = [{"text": "localization", "bbox": (0, 0, 50, 10), "size": 10},
             {"text": "mask", "bbox": (53, 0, 70, 10), "size": 10},      # gap of 3pt -> word break
             {"text": "s", "bbox": (70.2, 0, 73, 10), "size": 10},       # touching -> same word
             {"text": " ", "bbox": (73, 0, 75, 10), "size": 10},
             {"text": "and", "bbox": (75, 0, 90, 10), "size": 10}]
    assert _join_spans(spans) == "localization masks and"
