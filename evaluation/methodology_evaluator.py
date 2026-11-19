"""Methodological soundness dimension."""
from __future__ import annotations

import re

from evaluation.rubric import DimensionSpec, clamp


def signals(doc, analyzed, evidence, external) -> dict:
    method = doc.section_text("Methodology")
    return {
        "method_section_words": len(method.split()),
        "has_formalism": bool(re.search(r"(=|\\sum|∑|argmin|argmax|theorem|lemma|proof|loss)", method, re.I)),
        "states_assumptions": bool(re.search(r"\bassum(e|es|ption)", doc_text(doc), re.I)),
        "baselines_named": len(analyzed.profile.baselines),
        "sections_found": analyzed.sections_found,
    }


def doc_text(doc) -> str:
    return "\n".join(s.text for s in doc.sections if s.name not in ("References", "Front Matter"))


def fallback_score(s: dict) -> float:
    score = 4.0
    score += min(s["method_section_words"] / 600, 3)
    score += 1.0 if s["has_formalism"] else 0
    score += 0.5 if s["states_assumptions"] else 0
    score += 0.5 if s["baselines_named"] else 0
    return clamp(score)


SPEC = DimensionSpec(
    name="methodology", label="Methodological Soundness",
    questions=["Is the methodology appropriate for the research problem?", "Are assumptions justified?",
               "Are the algorithms suitable?", "Are the baselines appropriate?",
               "Is the argument coherent from problem to method to evaluation?"],
    signals=signals, fallback_score=fallback_score,
)
