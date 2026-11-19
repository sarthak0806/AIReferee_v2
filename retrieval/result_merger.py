"""Merge internal full-text passages with external metadata-only results.

Full-text passages (from FAISS/SQLite) and metadata-only results (arXiv
abstracts, Scholar snippets) are kept in separate lists; a snippet is never
promoted to a full-text evidence passage. External works that are already in
the internal corpus are linked to it.
"""
from __future__ import annotations

from database.db_manager import DatabaseManager
from retrieval.deduplicator import deduplicate, same_work


def merge_internal(per_query_hits: list[list[dict]], queries: list[dict]) -> list[dict]:
    """Flatten per-query FAISS hits; a chunk hit by several queries keeps its best score."""
    best: dict[str, dict] = {}
    for q, hits in zip(queries, per_query_hits):
        for h in hits:
            cur = best.get(h["chunk_id"])
            matched = cur["matched_queries"] if cur else set()
            if cur is None or h["similarity_score"] > cur["similarity_score"]:
                best[h["chunk_id"]] = {**h, "query": q["query"], "query_category": q.get("category"),
                                       "query_id": q.get("query_id"), "content_type": "full_text",
                                       "matched_queries": matched}
            best[h["chunk_id"]]["matched_queries"].add(q["query"])
    out = sorted(best.values(), key=lambda h: h["similarity_score"], reverse=True)
    for h in out:
        h["matched_queries"] = sorted(h["matched_queries"])
    return out


def merge_external(arxiv_results: list[dict], scholar_results: list[dict], db: DatabaseManager) -> list[dict]:
    merged = deduplicate([*arxiv_results, *scholar_results])
    corpus = db.conn.execute("SELECT paper_id, title, doi, source_url, conference FROM papers "
                             "WHERE corpus='reference' AND download_status='valid'").fetchall()
    for r in merged:
        for p in corpus:
            arx = p["source_url"].rsplit("/", 1)[-1] if "arxiv.org" in p["source_url"] else None
            if same_work(r, {"title": p["title"], "doi": p["doi"], "arxiv_id": arx}):
                r["in_internal_corpus"] = p["paper_id"]
                r["corpus_conference"] = p["conference"]
                break
        r["text_for_rerank"] = " ".join(x for x in (r.get("title"), r.get("abstract") or r.get("snippet")) if x)
    return merged
