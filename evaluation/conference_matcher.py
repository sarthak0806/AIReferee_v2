"""Conference selection for papers classified as Publishable.

Signals per conference (each min-max normalised across venues, then weighted):
  centroid    cosine(paper embedding, conference centroid)
  neighbours  share of FAISS nearest-neighbour similarity mass from that venue's papers
  keywords    TF-IDF weight of the venue's distinctive terms found in the paper
  llm         LLM fit judgement (0-10) given the venue profile and paper profile
Similarity is one signal, not the sole decision factor.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from pydantic import BaseModel, Field

from config import CONFIG, active_conferences
from utils.llm import LLMUnavailable, complete_json, llm_available
from utils.logging_config import get_logger

log = get_logger("conference_matcher")


class VenueFit(BaseModel):
    conference: str
    topic_alignment: float = Field(ge=0, le=10)
    methodology_alignment: float = Field(ge=0, le=10)
    contribution_alignment: float = Field(ge=0, le=10)
    overall: float = Field(ge=0, le=10)
    reason: str = ""


class VenueFits(BaseModel):
    fits: list[VenueFit]


LLM_PROMPT = """Rate how well the paper fits each venue (0-10) on topic, methodology and contribution alignment, and
overall suitability. Venue descriptions are derived only from their reference papers listed below.

<paper>
{paper}
</paper>

Venues:
{venues}"""


def _norm(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    return {k: (v - lo) / (hi - lo) if hi > lo else 0.5 for k, v in d.items()}


def keyword_score(text: str, keywords: dict[str, float]) -> float:
    low = text.lower()
    total = sum(keywords.values()) or 1.0
    hit = sum(w for k, w in keywords.items() if re.search(rf"\b{re.escape(k)}\b", low))
    return hit / total


def neighbour_votes(per_chunk_hits: list[list[dict]]) -> tuple[dict[str, float], dict[str, list[dict]]]:
    mass: dict[str, float] = defaultdict(float)
    papers: dict[str, dict[str, dict]] = defaultdict(dict)
    for hits in per_chunk_hits:
        for h in hits:
            conf = h.get("conference")
            mass[conf] += max(h["similarity_score"], 0)
            p = papers[conf].get(h["paper_id"])
            if p is None or h["similarity_score"] > p["similarity_score"]:
                papers[conf][h["paper_id"]] = {"paper_id": h["paper_id"], "title": h.get("title"),
                                               "year": h.get("publication_year"), "source_url": h.get("source_url"),
                                               "similarity_score": round(h["similarity_score"], 4),
                                               "section": h.get("section_name"), "passage": h["chunk_text"][:500]}
    total = sum(mass.values()) or 1.0
    shares = {c: m / total for c, m in mass.items()}
    top_papers = {c: sorted(v.values(), key=lambda x: x["similarity_score"], reverse=True)[:5] for c, v in papers.items()}
    return shares, top_papers


def llm_fit(profile_text: str, profiles: dict) -> dict[str, VenueFit]:
    venues = []
    for c, p in profiles.items():
        lp = p.get("llm_profile") or {}
        venues.append(f"- {c}: topics={lp.get('dominant_topics') or list(p['keywords'])[:12]}; "
                      f"methods={lp.get('frequent_methodologies')}; contributions={lp.get('typical_contributions')}; "
                      f"summary={lp.get('summary')}")
    out = complete_json(LLM_PROMPT.format(paper=profile_text[:6000], venues="\n".join(venues)), VenueFits,
                        max_tokens=4000)
    return {f.conference.lower().strip(): f for f in out.fits if f.conference.lower().strip() in profiles}


def match(paper_vec: np.ndarray, paper_text: str, profile_text: str, per_chunk_hits: list[list[dict]],
          profiles: dict, centroids: dict, use_llm: bool = True) -> dict:
    confs = [c for c in active_conferences() if c in profiles and c in centroids]
    centroid = {c: float(np.dot(paper_vec, centroids[c])) for c in confs}
    shares, top_papers = neighbour_votes(per_chunk_hits)
    neighbours = {c: shares.get(c, 0.0) for c in confs}
    keywords = {c: keyword_score(paper_text, profiles[c]["keywords"]) for c in confs}

    signals = {"centroid": _norm(centroid), "neighbours": _norm(neighbours), "keywords": _norm(keywords)}
    raw = {"centroid": centroid, "neighbours": neighbours, "keywords": keywords}
    fits: dict[str, VenueFit] = {}
    if use_llm and llm_available():
        try:
            fits = llm_fit(profile_text, {c: profiles[c] for c in confs})
            raw["llm"] = {c: fits[c].overall for c in confs if c in fits}
            signals["llm"] = {c: raw["llm"].get(c, 0) / 10 for c in confs}
        except LLMUnavailable as exc:
            log.warning("LLM venue fit unavailable: %s", exc)

    weights = {k: w for k, w in CONFIG["conference_signal_weights"].items() if k in signals}
    wsum = sum(weights.values())
    scores = {c: round(sum(weights[k] * signals[k][c] for k in weights) / wsum, 4) for c in confs}
    ranking = sorted(confs, key=lambda c: scores[c], reverse=True)
    return {
        "recommended": ranking[0] if ranking else None,
        "ranking": [{"conference": c, "score": scores[c],
                     "signals": {k: round(raw[k].get(c, 0), 4) for k in raw},
                     "llm_reason": fits[c].reason if c in fits else None,
                     "top_reference_papers": top_papers.get(c, [])} for c in ranking],
        "weights_used": weights,
    }
