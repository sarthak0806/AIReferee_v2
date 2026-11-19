"""End-to-end online assessment of one paper.

PDF -> structured profile -> LLM queries -> FAISS (bi-encoder) + arXiv + Scholar
-> merge/dedupe -> Cross-Encoder rerank -> evidence selection -> dimension
assessment -> publishability decision -> (if publishable) conference matching
and rationale. Every step is logged to SQLite and returned for the audit trail.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import CONFIG, path
from database.db_manager import DatabaseManager
from evaluation import conference_matcher, conference_profiler, publishability_classifier as clf
from evaluation.paper_analyzer import analyze_pdf
from evaluation.rationale_generator import conference_rationale
from ingestion.chunker import chunk_document
from retrieval.arxiv_search import search_arxiv
from retrieval.cross_encoder import get_reranker
from retrieval.evidence_selector import relevance_label, select_evidence
from retrieval.query_generator import generate_queries
from retrieval.result_merger import merge_external, merge_internal
from retrieval.scholar_search import scholar_available, search_scholar
from retrieval.similarity_search import SimilaritySearcher
from utils.logging_config import get_logger

log = get_logger("final_evaluator")

SCHOLAR_PRIORITY = ["novelty", "research_problem", "methodology", "dataset_experiments", "results", "limitations"]


def _profile_text(profile) -> str:
    p = profile.model_dump()
    return "\n".join(f"{k}: {v}" for k, v in p.items() if v)


def evaluate_paper(pdf_path: str | Path, paper_id: str | None = None, db: DatabaseManager | None = None,
                   use_external: bool = True, progress=None) -> dict:
    """Run the full pipeline; ``progress(step, message)`` is an optional UI callback."""
    own_db = db is None
    db = db or DatabaseManager()
    timings: dict[str, float] = {}
    report = lambda step, msg: (progress(step, msg) if progress else None, log.info("%s: %s", step, msg))
    t = time.perf_counter()

    # 1. Understand the paper ------------------------------------------------
    report("analyze", "Extracting text and building the research profile")
    analyzed, doc = analyze_pdf(pdf_path, paper_id)
    pid = analyzed.paper_id
    profile = analyzed.profile
    timings["analyze"] = time.perf_counter() - t

    # Leakage guard: if the paper itself is in the reference corpus, exclude it.
    same = db.find_by_sha(analyzed.sha256)
    exclude = {same["paper_id"]} if same else set()

    # 2. Queries ---------------------------------------------------------------
    t = time.perf_counter()
    report("queries", "Generating scholarly search queries")
    queries, query_source = generate_queries(profile)
    qdicts = []
    for q in queries:
        qid = db.insert_query(pid, q.query, q.category, q.reason, "faiss+arxiv+scholar")
        qdicts.append({"query_id": qid, "query": q.query, "category": q.category, "reason": q.reason})
    timings["queries"] = time.perf_counter() - t

    # 3. Stage 1: bi-encoder FAISS retrieval --------------------------------------
    t = time.perf_counter()
    report("retrieve", "Searching the internal FAISS index")
    searcher = SimilaritySearcher(db)
    per_query = searcher.search([q["query"] for q in qdicts], CONFIG["top_k_similarity"], exclude)
    timings["faiss"] = time.perf_counter() - t

    # 4. External discovery (metadata only) ------------------------------------------
    t = time.perf_counter()
    arxiv_res, scholar_res = [], []
    if use_external:
        report("external", "Searching arXiv and Google Scholar")
        for q in qdicts:
            arxiv_res += search_arxiv(q["query"], db)
        if scholar_available():
            ordered = sorted(qdicts, key=lambda q: SCHOLAR_PRIORITY.index(q["category"]))
            for q in ordered[:CONFIG["max_scholar_queries_per_paper"]]:
                scholar_res += search_scholar(q["query"], db)
    external = merge_external(arxiv_res, scholar_res, db)
    # Leakage guard: the search engines often return the submitted paper itself
    # (e.g. its arXiv preprint). It must not count as prior work.
    from retrieval.deduplicator import same_work
    self_ref = {"title": profile.title}
    self_hits = [x["title"] for x in external if same_work(x, self_ref)]
    external = [x for x in external if not same_work(x, self_ref)]
    timings["external"] = time.perf_counter() - t

    # 5. Stage 2: cross-encoder reranking ------------------------------------------
    t = time.perf_counter()
    report("rerank", "Cross-Encoder reranking")
    reranker = get_reranker()
    best: dict[str, dict] = {}
    log_rows = []
    for q, hits in zip(qdicts, per_query):
        ranked = reranker.rerank(q["query"], hits)
        for h in ranked:
            log_rows.append({"uploaded_paper_id": pid, "query_id": q["query_id"], "chunk_id": h["chunk_id"],
                             "similarity_score": h["similarity_score"], "reranker_score": h["reranker_score"],
                             "initial_rank": h["initial_rank"], "final_rank": h["final_rank"]})
            cur = best.get(h["chunk_id"])
            if cur is None or h["reranker_score"] > cur["reranker_score"]:
                best[h["chunk_id"]] = {**h, "query": q["query"], "query_category": q["category"]}
    db.insert_retrieval_logs(log_rows)
    ranked_all = sorted(best.values(), key=lambda h: h["reranker_score"], reverse=True)
    rerank_query = next((q["query"] for q in qdicts if q["category"] == "research_problem"), qdicts[0]["query"])
    if external:
        ext_scores = reranker.score([(rerank_query, x["text_for_rerank"]) for x in external])
        for x, s in zip(external, ext_scores):
            x["reranker_score"] = round(s, 4)
            x["relevance"] = relevance_label(s)
        external.sort(key=lambda x: x["reranker_score"], reverse=True)
    for i, x in enumerate(external, 1):
        x["evidence_id"] = f"X{i}"
    timings["rerank"] = time.perf_counter() - t

    # 6. Evidence selection --------------------------------------------------------
    evidence = select_evidence(ranked_all, CONFIG["final_evidence_count"])
    ext_for_llm = external[:8]

    # 7. Publishability assessment ---------------------------------------------------
    t = time.perf_counter()
    report("assess", "Assessing the six evaluation dimensions")
    signals = clf.compute_signals(doc, analyzed, evidence, external)
    insufficient = analyzed.char_count < 6000 or len(set(analyzed.sections_found) - {"Front Matter", "References"}) < 2
    assessment, assess_source = clf.assess(doc, signals, evidence, ext_for_llm)
    decision = clf.decide(assessment, assess_source, insufficient)
    timings["assess"] = time.perf_counter() - t

    eval_id = db.insert_evaluation({
        "uploaded_paper_id": pid,
        **{f"{d}_score": getattr(assessment, d).score for d in
           ("novelty", "methodology", "literature", "results", "reproducibility")},
        "experimental_score": assessment.experimental.score,
        "publishability_decision": decision.decision,
        "decision_rationale": assessment.overall_summary or "; ".join(decision.reasons),
        "confidence": decision.confidence,
    })

    # 8. Conference selection (publishable papers only) -------------------------------
    conference = None
    t = time.perf_counter()
    if decision.csv_publishable == 1:
        report("conference", "Matching against conference profiles")
        profiles, centroids = conference_profiler.load_profiles(db)
        chunks = chunk_document(doc, pid, None)
        texts = [c.chunk_text for c in chunks]
        step = max(1, len(texts) // CONFIG["conference_max_probe_chunks"])
        probe = texts[::step][:CONFIG["conference_max_probe_chunks"]]
        vecs = searcher.model.encode_documents(probe)
        pvec = vecs.mean(axis=0)
        pvec = pvec / (np.linalg.norm(pvec) + 1e-12)
        hits = searcher.search_vectors(vecs, CONFIG["conference_neighbour_k"], exclude)
        ptext = _profile_text(profile)
        m = conference_matcher.match(pvec, " ".join(texts), ptext, hits, profiles, centroids)
        conf = m["recommended"]
        rationale, rat_source = conference_rationale(conf, ptext, profiles[conf], m["ranking"][0])
        conf_id = db.upsert_conference(conf, conf.upper())
        db.store_recommendation(eval_id, conf_id, m["ranking"][0]["score"], rationale)
        conference = {**m, "rationale": rationale, "rationale_source": rat_source}
    timings["conference"] = time.perf_counter() - t

    result = {
        "paper_id": pid,
        "pdf_name": Path(pdf_path).name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": analyzed.model_dump(),
        "excluded_self_matches": sorted(exclude),
        "excluded_external_self_matches": self_hits,
        "queries": qdicts, "query_source": query_source,
        "internal_results": [[{k: h[k] for k in ("chunk_id", "title", "conference", "section_name", "page_number",
                                                  "similarity_score", "initial_rank")} for h in hits]
                             for hits in per_query],
        "external_results": external,
        "evidence": evidence,
        "signals": signals,
        "assessment": assessment.model_dump(),
        "decision": decision.model_dump(),
        "evaluation_id": eval_id,
        "conference": conference,
        "config": {k: CONFIG[k] for k in ("embedding_model", "reranker_model", "llm_model", "top_k_similarity",
                                          "final_evidence_count", "dimension_weights", "publishable_threshold",
                                          "human_review_csv_policy", "chunk_size", "chunk_overlap")},
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
    }
    out_dir = path("reports_dir") / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if own_db:
        db.close()
    return result


def csv_row(result: dict) -> dict:
    """Hackathon CSV row from a result."""
    d = result["decision"]
    if d["csv_publishable"] == 1 and result.get("conference"):
        return {"Paper ID": result["paper_id"], "Publishable": 1,
                "Conference": result["conference"]["recommended"], "Rationale": result["conference"]["rationale"]}
    return {"Paper ID": result["paper_id"], "Publishable": 0, "Conference": "na", "Rationale": "na"}
