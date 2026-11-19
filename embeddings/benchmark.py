"""Retrieval, reranking and conference-matching benchmarks on the reference corpus.

No human relevance judgements exist for this corpus, so retrieval is measured
with a reproducible *known-item* proxy:
  query    = a paper's title (short, search-like) or its official abstract
  relevant = chunks of that same paper (Abstract-section chunks are removed from
             the candidate pool so the abstract query cannot match itself)
Metrics are computed at paper level after collapsing ranked chunks to papers
(Recall@K = hit rate, MRR, nDCG@10) plus chunk-level Precision@K (share of the
top-K chunks that come from the target paper). This measures whether the
embedding finds the right paper's body text; it is not a substitute for expert
relevance labels.

Conference selection is measured with leave-one-out over the reference papers:
each paper is held out of the centroids, keyword profiles and neighbour search.

Usage:
    python -m embeddings.benchmark retrieval [--models all]
    python -m embeddings.benchmark rerank
    python -m embeddings.benchmark conference
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np

from config import CONFIG, active_conferences, path
from database.db_manager import DatabaseManager
from embeddings.embedding_generator import embed_pending
from embeddings.embedding_model import get_embedding_model
from retrieval.faiss_manager import FaissManager
from utils.logging_config import get_logger

log = get_logger("benchmark")

KS = (5, 10)


# ---------------------------------------------------------------- metrics
def paper_ranking(hits: list[dict]) -> list[str]:
    seen, out = set(), []
    for h in hits:
        if h["paper_id"] not in seen:
            seen.add(h["paper_id"])
            out.append(h["paper_id"])
    return out


def metrics_for(runs: list[tuple[str, list[dict]]]) -> dict:
    """runs: (target_paper_id, ranked chunk hits)."""
    agg = defaultdict(list)
    for target, hits in runs:
        papers = paper_ranking(hits)
        rank = papers.index(target) + 1 if target in papers else None
        for k in KS:
            agg[f"recall@{k}"].append(1.0 if rank and rank <= k else 0.0)
            top = hits[:k]
            agg[f"precision@{k}"].append(sum(h["paper_id"] == target for h in top) / k)
        agg["mrr"].append(1.0 / rank if rank else 0.0)
        agg["ndcg@10"].append(1.0 / math.log2(rank + 1) if rank and rank <= 10 else 0.0)
    return {k: round(float(np.mean(v)), 4) for k, v in agg.items()} | {"queries": len(runs)}


def _queries(db: DatabaseManager, kind: str) -> list[tuple[str, str]]:
    rows = db.conn.execute("SELECT paper_id, title, abstract FROM papers WHERE corpus='reference' "
                           "AND download_status='valid' AND extraction_status='extracted' ORDER BY paper_id").fetchall()
    return [(r[0], r[1] if kind == "title" else (r[2] or r[1])) for r in rows]


def _abstract_chunk_ids(db: DatabaseManager) -> set[str]:
    return {r[0] for r in db.conn.execute("SELECT chunk_id FROM chunks WHERE section_name='Abstract'")}


def _search(fm: FaissManager, qvecs: np.ndarray, k: int, drop: set[str]) -> list[list[dict]]:
    res = fm.search(qvecs, k + 40)
    return [[h for h in hits if h["chunk_id"] not in drop][:k] for hits in res]


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:
        return None


# --------------------------------------------------------------- retrieval
def benchmark_retrieval(models: list[str]) -> dict:
    db = DatabaseManager()
    drop = _abstract_chunk_ids(db)
    out = {}
    for name in models:
        rss0 = _rss_mb()
        try:
            build = embed_pending(db, name)
        except Exception as exc:  # model download / remote-code failures are reported, not hidden
            log.error("Model %s failed: %s", name, exc)
            out[name] = {"error": str(exc)[:300]}
            continue
        model = get_embedding_model(name)
        fm = FaissManager(model.dim, model.name, db).load_or_create()
        res = {"dim": model.dim, "index_vectors": fm.ntotal,
               "index_build_seconds_new_vectors": build["seconds"], "new_vectors": build["new_vectors"]}
        # Embedding latency on a fixed sample of 256 chunks.
        sample = [r[0] for r in db.conn.execute("SELECT chunk_text FROM chunks ORDER BY chunk_id LIMIT 256")]
        t = time.perf_counter()
        model.encode_documents(sample)
        res["doc_embed_ms_per_chunk"] = round((time.perf_counter() - t) * 1000 / len(sample), 2)
        for kind in ("title", "abstract"):
            qs = _queries(db, kind)
            t = time.perf_counter()
            qv = model.encode_queries([q for _, q in qs])
            q_ms = (time.perf_counter() - t) * 1000 / len(qs)
            t = time.perf_counter()
            hits = _search(fm, qv, 30, drop)
            s_ms = (time.perf_counter() - t) * 1000 / len(qs)
            res[kind] = metrics_for([(pid, h) for (pid, _), h in zip(qs, hits)]) | {
                "query_embed_ms": round(q_ms, 2), "search_ms": round(s_ms, 2)}
        rss1 = _rss_mb()
        res["params_millions"] = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1)
        res["process_rss_mb_after"] = round(rss1, 0) if rss1 else None
        res["rss_increase_mb"] = round(rss1 - rss0, 0) if rss0 and rss1 else None
        res["index_file_mb"] = round(fm.index_file.stat().st_size / 2**20, 2) if fm.index_file.exists() else None
        out[name] = res
        log.info("%s: %s", name, json.dumps({k: res[k] for k in ("title", "abstract")}))
    db.close()
    return out


# ----------------------------------------------------------------- rerank
def benchmark_rerank(rerankers: list[str], k_first: int = 30) -> dict:
    from retrieval.cross_encoder import Reranker

    db = DatabaseManager()
    drop = _abstract_chunk_ids(db)
    model = get_embedding_model()
    fm = FaissManager(model.dim, model.name, db).load_or_create()
    out = {}
    for kind in ("title", "abstract"):
        qs = _queries(db, kind)
        first = _search(fm, model.encode_queries([q for _, q in qs]), k_first, drop)
        out[kind] = {"bi_encoder_only": metrics_for([(pid, h) for (pid, _), h in zip(qs, first)])}
        out[kind]["first_stage_recall@30"] = round(float(np.mean(
            [pid in {h["paper_id"] for h in h30} for (pid, _), h30 in zip(qs, first)])), 4)
        for name in rerankers:
            try:
                rr = Reranker(name)
            except Exception as exc:
                out[kind][name] = {"error": str(exc)[:300]}
                continue
            t = time.perf_counter()
            reranked = [rr.rerank(q, hits) for (_, q), hits in zip(qs, first)]
            ms = (time.perf_counter() - t) * 1000 / len(qs)
            out[kind][name] = metrics_for([(pid, h) for (pid, _), h in zip(qs, reranked)]) | {
                "rerank_ms_per_query": round(ms, 1)}
        log.info("rerank %s: %s", kind, json.dumps(out[kind]))
    db.close()
    return out


# ------------------------------------------------------------- conference
def benchmark_conference() -> dict:
    """Leave-one-out conference prediction with the non-LLM signals."""
    from evaluation.conference_matcher import _norm, keyword_score, neighbour_votes
    from evaluation.conference_profiler import paper_texts, paper_vectors, venue_keywords

    db = DatabaseManager()
    model = get_embedding_model()
    fm = FaissManager(model.dim, model.name, db).load_or_create()
    confs = [c for c in active_conferences() if db.count_valid(c)]
    texts, labels, pids = [], [], []
    for c in confs:
        for pid, txt in paper_texts(db, c).items():
            pids.append(pid)
            texts.append(txt)
            labels.append(c)
    pvecs = paper_vectors(db, fm, pids)
    weights = {k: v for k, v in CONFIG["conference_signal_weights"].items() if k != "llm"}
    preds = {name: [] for name in ("centroid", "neighbours", "keywords", "combined")}
    for i, pid in enumerate(pids):
        rest = [j for j in range(len(pids)) if j != i]
        cents = {}
        for c in confs:
            vs = [pvecs[pids[j]] for j in rest if labels[j] == c and pids[j] in pvecs]
            m = np.mean(vs, axis=0)
            cents[c] = m / np.linalg.norm(m)
        kw = venue_keywords([texts[j] for j in rest], [labels[j] for j in rest])
        vids = [r[0] for r in db.conn.execute(
            "SELECT v.vector_id FROM vector_map v JOIN chunks c ON c.chunk_id=v.chunk_id "
            "WHERE c.paper_id=? AND v.embedding_model=?", (pid, model.name))]
        step = max(1, len(vids) // CONFIG["conference_max_probe_chunks"])
        probe = fm.reconstruct(vids[::step][:CONFIG["conference_max_probe_chunks"]])
        hits = fm.search(probe, CONFIG["conference_neighbour_k"], exclude_paper_ids={pid})
        shares, _ = neighbour_votes(hits)
        raw = {"centroid": {c: float(np.dot(pvecs[pid], cents[c])) for c in confs},
               "neighbours": {c: shares.get(c, 0.0) for c in confs},
               "keywords": {c: keyword_score(texts[i], kw[c]) for c in confs}}
        for name in ("centroid", "neighbours", "keywords"):
            preds[name].append(max(raw[name], key=raw[name].get))
        normed = {k: _norm(v) for k, v in raw.items()}
        comb = {c: sum(weights[k] * normed[k][c] for k in weights) for c in confs}
        preds["combined"].append(max(comb, key=comb.get))
    out = {"papers": len(pids), "signals_used": list(weights), "note": "leave-one-out; LLM signal excluded"}
    for name, p in preds.items():
        acc = float(np.mean([a == b for a, b in zip(p, labels)]))
        cm = {t: dict(Counter(pp for pp, tt in zip(p, labels) if tt == t)) for t in confs}
        out[name] = {"accuracy": round(acc, 4), "confusion": cm}
    db.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "rerank", "conference", "all"])
    ap.add_argument("--models", default="default", help="'default', 'all' or comma-separated model ids")
    args = ap.parse_args()
    models = (CONFIG["benchmark_embedding_models"] if args.models == "all" else
              [CONFIG["embedding_model"]] if args.models == "default" else args.models.split(","))
    results = {}
    if args.what in ("retrieval", "all"):
        results["retrieval"] = benchmark_retrieval(models)
    if args.what in ("rerank", "all"):
        results["rerank"] = benchmark_rerank(CONFIG["benchmark_reranker_models"])
    if args.what in ("conference", "all"):
        results["conference"] = benchmark_conference()
    out = path("reports_dir") / f"benchmark_{args.what}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
