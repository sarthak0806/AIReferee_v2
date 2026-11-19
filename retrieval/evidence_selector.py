"""Select a diverse, deduplicated evidence set from reranked passages."""
from __future__ import annotations

import json

from rapidfuzz import fuzz

from config import CONFIG


def relevance_label(score: float) -> str:
    t = CONFIG["relevance_thresholds"]
    if score >= t["direct"]:
        return "directly relevant"
    if score >= t["indirect"]:
        return "indirectly relevant"
    return "weakly relevant"


def _authors(a) -> list[str]:
    if isinstance(a, list):
        return a
    try:
        return json.loads(a) if a else []
    except (TypeError, json.JSONDecodeError):
        return []


def select_evidence(ranked: list[dict], n: int | None = None, max_per_paper: int | None = None) -> list[dict]:
    """``ranked`` must be sorted by reranker score. Applies:
    duplicate-passage removal, a per-paper cap, and a diversity pass that first
    takes the best passage of distinct papers before taking second passages."""
    n = n or CONFIG["final_evidence_count"]
    max_per_paper = max_per_paper or CONFIG["max_evidence_per_paper"]
    unique: list[dict] = []
    for c in ranked:
        if any(c["chunk_id"] == u["chunk_id"] or
               fuzz.ratio(c["chunk_text"][:400], u["chunk_text"][:400]) > 90 for u in unique):
            continue
        unique.append(c)

    chosen: list[dict] = []
    per_paper: dict[str, int] = {}
    for round_cap in range(1, max_per_paper + 1):  # round 1 = one per paper (diversity)
        for c in unique:
            if len(chosen) >= n:
                break
            if c in chosen or per_paper.get(c["paper_id"], 0) >= round_cap:
                continue
            chosen.append(c)
            per_paper[c["paper_id"]] = per_paper.get(c["paper_id"], 0) + 1
    chosen.sort(key=lambda c: c["reranker_score"], reverse=True)

    evidence = []
    for i, c in enumerate(chosen, 1):
        evidence.append({
            "evidence_id": f"E{i}",
            "chunk_id": c["chunk_id"], "paper_id": c["paper_id"],
            "title": c.get("title"), "authors": _authors(c.get("authors")),
            "conference": c.get("conference"), "year": c.get("publication_year"),
            "section": c.get("section_name"), "page": c.get("page_number"),
            "passage": c["chunk_text"],
            "similarity_score": round(c["similarity_score"], 4),
            "reranker_score": round(c["reranker_score"], 4),
            "relevance": relevance_label(c["reranker_score"]),
            "doi": c.get("doi"), "source_url": c.get("source_url"),
            "query": c.get("query"), "query_category": c.get("query_category"),
            "content_type": "full_text",
        })
    return evidence
