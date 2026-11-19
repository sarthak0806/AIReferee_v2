"""Novelty dimension: distinctness of the contribution from the closest prior work.

Similarity is a signal of overlap, never proof of plagiarism or lack of novelty.
"""
from __future__ import annotations

from evaluation.rubric import DimensionSpec, clamp


def signals(doc, analyzed, evidence: list[dict], external: list[dict]) -> dict:
    sims = [e["similarity_score"] for e in evidence]
    rr = [e["reranker_score"] for e in evidence]
    return {
        "claimed_contributions": len(analyzed.profile.claimed_contributions),
        "max_internal_similarity": round(max(sims), 3) if sims else None,
        "max_reranker_score": round(max(rr), 2) if rr else None,
        "directly_relevant_prior_passages": sum(e["relevance"] == "directly relevant" for e in evidence),
        "highly_related_external_works": sum((x.get("reranker_score") or -99) > 2 for x in external),
    }


def fallback_score(s: dict) -> float:
    score = 6.0
    score += 0.5 if s["claimed_contributions"] >= 2 else -1.0
    if (s["max_internal_similarity"] or 0) > 0.85:
        score -= 1.5  # near-duplicate prior passage: flag for review, not a verdict
    score -= 0.3 * min(s["highly_related_external_works"], 5)
    return clamp(score)


SPEC = DimensionSpec(
    name="novelty", label="Novelty",
    questions=["Is the contribution distinct from the closest prior work in the evidence?",
               "Are similar approaches already established?", "Is the research gap justified?",
               "Are the claimed improvements meaningful?"],
    signals=signals, fallback_score=fallback_score,
)
