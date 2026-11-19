"""Uploaded-paper understanding: PDF -> sections -> validated research profile."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, Field

from config import CONFIG
from ingestion.extractor import ExtractedDocument, extract_pdf
from utils.llm import LLMUnavailable, complete_json, llm_available
from utils.logging_config import get_logger

log = get_logger("paper_analyzer")


class ResearchProfile(BaseModel):
    title: str = ""
    abstract: str = ""
    research_problem: str = ""
    research_objectives: list[str] = Field(default_factory=list)
    claimed_contributions: list[str] = Field(default_factory=list)
    methodology: str = ""
    datasets: list[str] = Field(default_factory=list)
    experimental_design: str = ""
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    results: str = ""
    conclusion: str = ""
    limitations: list[str] = Field(default_factory=list)
    research_area_keywords: list[str] = Field(default_factory=list)


class AnalyzedPaper(BaseModel):
    paper_id: str
    sha256: str
    profile: ResearchProfile
    profile_source: str          # groq:<model> | heuristic-fallback
    page_count: int
    sections_found: list[str]
    char_count: int
    reference_count: int
    model_config = {"arbitrary_types_allowed": True}


def _take(doc: ExtractedDocument, names: list[str], limit: int) -> str:
    text = "\n".join(doc.section_text(n) for n in names if doc.section_text(n))
    return text[:int(limit)]


def build_llm_input(doc: ExtractedDocument) -> str:
    """Section-balanced excerpt that fits the LLM budget."""
    budget = CONFIG["llm_max_input_chars"]
    parts = [
        ("TITLE", doc.title, 400),
        ("ABSTRACT", doc.abstract, 3000),
        ("INTRODUCTION", _take(doc, ["Introduction"], 0.18 * budget), 0),
        ("METHOD", _take(doc, ["Methodology"], 0.25 * budget), 0),
        ("EXPERIMENTS", _take(doc, ["Experiments"], 0.18 * budget), 0),
        ("RESULTS", _take(doc, ["Results", "Discussion"], 0.12 * budget), 0),
        ("CONCLUSION/LIMITATIONS", _take(doc, ["Conclusion", "Limitations"], 0.08 * budget), 0),
    ]
    out = []
    for label, text, cap in parts:
        if text:
            out.append(f"## {label}\n{text[:int(cap) if cap else None]}")
    used = sum(len(o) for o in out)
    if used < 0.5 * budget:  # weak sectioning: add raw body text
        body = "\n".join(s.text for s in doc.sections if s.name not in ("References", "Front Matter"))
        out.append(f"## BODY\n{body[: int(budget - used)]}")
    return "\n\n".join(out)[:budget]


PROFILE_PROMPT = """Extract a structured research profile from the paper below. Use only what the paper states.
Keep each field concise (<= 80 words; list items <= 20 words). Use empty strings/lists when the paper does not say.
"limitations" must contain limitations the paper itself acknowledges.

<paper>
{paper}
</paper>"""


def heuristic_profile(doc: ExtractedDocument) -> ResearchProfile:
    """No-LLM fallback: fields filled from detected sections only (no inference)."""
    def first_sents(text: str, n: int = 3) -> str:
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return " ".join(sents[:n])[:800]

    intro = doc.section_text("Introduction")
    contrib = re.findall(r"(?:we propose|we introduce|we present|our contributions?|we show)[^.]{10,250}\.", intro, re.I)
    exp = doc.section_text("Experiments")
    datasets = sorted(set(re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:-\d+)?)\s+(?:dataset|benchmark)", exp)))[:10]
    return ResearchProfile(
        title=doc.title, abstract=doc.abstract[:3000],
        research_problem=first_sents(intro, 3),
        claimed_contributions=[c.strip() for c in contrib[:5]],
        methodology=first_sents(doc.section_text("Methodology"), 4),
        datasets=datasets,
        experimental_design=first_sents(exp, 3),
        results=first_sents(doc.section_text("Results") or exp[-1500:], 3),
        conclusion=first_sents(doc.section_text("Conclusion"), 3),
        limitations=[first_sents(doc.section_text("Limitations"), 2)] if doc.section_text("Limitations") else [],
    )


def analyze_pdf(pdf_path: str | Path, paper_id: str | None = None) -> tuple[AnalyzedPaper, ExtractedDocument]:
    pdf_path = Path(pdf_path)
    data = pdf_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    doc = extract_pdf(pdf_path)
    profile_source = "heuristic-fallback"
    profile = None
    if llm_available():
        try:
            profile = complete_json(PROFILE_PROMPT.format(paper=build_llm_input(doc)), ResearchProfile)
            from utils.llm import model_tag
            profile_source = model_tag()
        except LLMUnavailable as exc:
            log.warning("Profile extraction fell back to heuristics: %s", exc)
    if profile is None:
        profile = heuristic_profile(doc)
    if not profile.title:
        profile.title = doc.title
    if not profile.abstract:
        profile.abstract = doc.abstract[:3000]
    refs = doc.section_text("References")
    ref_count = len(re.findall(r"(?:^|\n|\s)(?:\[\d+\]|\d+\.\s+[A-Z])", refs)) or refs.count(". 20") + refs.count(". 19")
    analyzed = AnalyzedPaper(
        paper_id=paper_id or f"upload_{sha[:10]}", sha256=sha, profile=profile, profile_source=profile_source,
        page_count=doc.page_count, sections_found=sorted({s.name for s in doc.sections}),
        char_count=doc.char_count, reference_count=ref_count,
    )
    return analyzed, doc
