"""Deduplicate external search results by DOI, arXiv id, normalised and fuzzy title."""
from __future__ import annotations

from rapidfuzz import fuzz

from config import CONFIG
from utils.text_utils import normalize_title


def _merge(into: dict, other: dict) -> None:
    into.setdefault("sources", [into.get("source")])
    if other.get("source") not in into["sources"]:
        into["sources"].append(other.get("source"))
    for k in ("doi", "arxiv_id", "abstract", "pdf_url", "year", "cited_by", "snippet"):
        if not into.get(k) and other.get(k):
            into[k] = other[k]
    if len(other.get("authors") or []) > len(into.get("authors") or []):
        into["authors"] = other["authors"]
    qs = set(into.get("queries", [into.get("query")])) | {other.get("query")}
    into["queries"] = sorted(q for q in qs if q)


def same_work(a: dict, b: dict, threshold: int | None = None) -> bool:
    threshold = threshold or CONFIG["fuzzy_title_threshold"]
    if a.get("doi") and b.get("doi") and a["doi"].lower() == b["doi"].lower():
        return True
    if a.get("arxiv_id") and b.get("arxiv_id") and a["arxiv_id"] == b["arxiv_id"]:
        return True
    ta, tb = normalize_title(a.get("title", "")), normalize_title(b.get("title", ""))
    if not ta or not tb:
        return False
    return ta == tb or fuzz.token_sort_ratio(ta, tb) >= threshold


def deduplicate(results: list[dict]) -> list[dict]:
    unique: list[dict] = []
    for r in results:
        if not r.get("title"):
            continue
        for u in unique:
            if same_work(u, r):
                _merge(u, r)
                break
        else:
            item = dict(r)
            item["sources"] = [r.get("source")]
            item["queries"] = [r.get("query")] if r.get("query") else []
            unique.append(item)
    return unique
