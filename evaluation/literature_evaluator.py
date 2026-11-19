"""Literature coverage dimension.

Checks whether highly relevant works found by retrieval appear in the paper's
own reference list (fuzzy title match). An uncited relevant work is reported
as *missing evidence to consider*, not as an error by itself.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from evaluation.rubric import DimensionSpec, clamp
from utils.text_utils import normalize_title


def cited(title: str, references_text: str) -> bool:
    t = normalize_title(title)
    if len(t) < 12:
        return False
    refs = normalize_title(references_text)
    return t in refs or fuzz.partial_ratio(t, refs) >= 90


def signals(doc, analyzed, evidence, external) -> dict:
    refs = doc.section_text("References")
    relevant = [x for x in external if (x.get("reranker_score") or -99) > 0]
    relevant += [{"title": e["title"]} for e in evidence if e["relevance"] == "directly relevant"]
    seen, uncited = set(), []
    for r in relevant:
        t = r.get("title") or ""
        if not t or normalize_title(t) in seen or normalize_title(t) == normalize_title(analyzed.profile.title):
            continue
        seen.add(normalize_title(t))
        if not cited(t, refs):
            uncited.append(t)
    return {
        "reference_count": analyzed.reference_count,
        "has_related_work_section": "Related Work" in analyzed.sections_found,
        "relevant_retrieved_works": len(seen),
        "relevant_works_not_cited": uncited[:8],
    }


def fallback_score(s: dict) -> float:
    score = 3.0 + min(s["reference_count"] / 10, 3.5)
    score += 1.5 if s["has_related_work_section"] else 0
    if s["relevant_retrieved_works"]:
        score -= 2.0 * len(s["relevant_works_not_cited"]) / s["relevant_retrieved_works"]
    return clamp(score)


SPEC = DimensionSpec(
    name="literature", label="Literature Coverage",
    questions=["Is relevant prior work acknowledged?", "Are important competing approaches missing?",
               "Are research-gap claims justified by the literature?"],
    signals=signals, fallback_score=fallback_score,
)
