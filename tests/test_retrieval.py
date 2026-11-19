import pytest

from retrieval.arxiv_search import parse_feed
from retrieval.deduplicator import deduplicate
from retrieval.evidence_selector import relevance_label, select_evidence
from retrieval.query_generator import ScholarlyQuery, _dedupe
from retrieval.result_merger import merge_internal
from retrieval.scholar_search import parse_results

FEED = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2401.00001v2</id><title>Graph  Models
for Traffic</title><summary>We study traffic.</summary><published>2024-01-02T00:00:00Z</published>
<author><name>A. One</name></author><author><name>B. Two</name></author>
<link title="pdf" href="https://arxiv.org/pdf/2401.00001v2"/><arxiv:doi>10.1/xyz</arxiv:doi></entry></feed>"""


def test_parse_arxiv_feed():
    r = parse_feed(FEED)[0]
    assert r["arxiv_id"] == "2401.00001" and r["title"] == "Graph Models for Traffic"
    assert r["authors"] == ["A. One", "B. Two"] and r["year"] == 2024 and r["content_type"] == "metadata_only"


def test_parse_scholar_payload():
    payload = {"organic_results": [{
        "title": "Attention Is All You Need", "link": "https://arxiv.org/abs/1706.03762",
        "snippet": "The dominant sequence transduction models...",
        "publication_info": {"summary": "A Vaswani, N Shazeer - Advances in neural …, 2017 - proceedings.neurips.cc",
                             "authors": [{"name": "A Vaswani"}, {"name": "N Shazeer"}]},
        "inline_links": {"cited_by": {"total": 100000}}}]}
    r = parse_results(payload, "q")[0]
    assert r["year"] == 2017 and r["arxiv_id"] == "1706.03762" and r["cited_by"] == 100000
    assert r["content_type"] == "metadata_only" and r["abstract"] is None


def test_deduplicate_by_id_and_fuzzy_title():
    res = deduplicate([
        {"source": "arxiv", "title": "Attention Is All You Need", "arxiv_id": "1706.03762", "query": "a"},
        {"source": "google_scholar", "title": "Attention is all you need.", "cited_by": 5, "query": "b"},
        {"source": "google_scholar", "title": "Something Else", "doi": "10.1/x", "query": "a"},
        {"source": "arxiv", "title": "Different title same doi", "doi": "10.1/X", "query": "c"},
    ])
    assert len(res) == 2
    assert set(res[0]["sources"]) == {"arxiv", "google_scholar"} and res[0]["cited_by"] == 5
    assert res[0]["queries"] == ["a", "b"]


def _hit(cid, pid, sim, rr=None, text=None):
    return {"chunk_id": cid, "paper_id": pid, "similarity_score": sim, "reranker_score": rr if rr is not None else sim,
            "chunk_text": text or TEXTS.get(cid, f"Passage about topic {cid}."), "title": pid, "authors": '["X"]'}


TEXTS = {"a1": "Graph neural networks forecast traffic on road sensor networks with attention.",
         "a2": "We ablate the temporal convolution kernel size and report error on METR-LA.",
         "a3": "Training uses Adam with learning rate 0.001 for one hundred epochs on four GPUs.",
         "b1": "Diffusion models generate textured meshes from a single input photograph.",
         "c1": "Multilingual corpora for low-resource languages suffer from noisy web crawls.",
         "dup": "Graph neural networks forecast traffic on road sensor networks with attention."}


def test_merge_internal_keeps_best_score():
    q = [{"query": "q1", "category": "methodology", "query_id": 1}, {"query": "q2", "category": "novelty", "query_id": 2}]
    merged = merge_internal([[_hit("c1", "p1", 0.5)], [_hit("c1", "p1", 0.7), _hit("c2", "p2", 0.6)]], q)
    assert [m["chunk_id"] for m in merged] == ["c1", "c2"]
    assert merged[0]["similarity_score"] == 0.7 and merged[0]["matched_queries"] == ["q1", "q2"]
    assert all(m["content_type"] == "full_text" for m in merged)


def test_evidence_selection_diversity_and_caps():
    ranked = [_hit("a1", "A", .9, 5), _hit("a2", "A", .8, 4.5), _hit("a3", "A", .7, 4), _hit("b1", "B", .6, 3),
              _hit("c1", "C", .5, 1), _hit("dup", "D", .5, 0.5, text=_hit("a1", "A", 0)["chunk_text"])]
    ev = select_evidence(ranked, n=4, max_per_paper=2)
    ids = [e["chunk_id"] for e in ev]
    assert "dup" not in ids                         # near-duplicate passage removed
    assert ids.count("a1") + ids.count("a2") + ids.count("a3") <= 2   # per-paper cap
    assert {"b1", "c1"} <= set(ids)                 # diversity: other papers included before 2nd A passage
    assert [e["reranker_score"] for e in ev] == sorted([e["reranker_score"] for e in ev], reverse=True)
    for k in ("title", "authors", "section", "page", "passage", "similarity_score", "reranker_score", "relevance"):
        assert k in ev[0]
    assert ev[0]["authors"] == ["X"]


def test_relevance_labels():
    assert relevance_label(5) == "directly relevant"
    assert relevance_label(0) == "indirectly relevant"
    assert relevance_label(-5) == "weakly relevant"


def test_query_dedupe():
    qs = [ScholarlyQuery(query="graph neural network traffic forecasting", category="methodology"),
          ScholarlyQuery(query="traffic forecasting graph neural network", category="novelty"),
          ScholarlyQuery(query="spatio temporal benchmark datasets metrics", category="dataset_experiments")]
    assert len(_dedupe(qs)) == 2


@pytest.mark.corpus
def test_retrieval_metrics_on_corpus():
    from embeddings.benchmark import benchmark_retrieval
    from config import CONFIG
    res = benchmark_retrieval([CONFIG["embedding_model"]])[CONFIG["embedding_model"]]
    for kind in ("title", "abstract"):
        m = res[kind]
        for k in ("recall@5", "recall@10", "precision@5", "precision@10", "mrr", "ndcg@10"):
            assert 0.0 <= m[k] <= 1.0
        assert m["recall@10"] >= m["recall@5"]
        assert m["recall@10"] > 0.5   # known-item sanity floor: the right paper is usually found


def test_self_match_detected_for_external_results():
    from retrieval.deduplicator import same_work
    me = {"title": "DisCoScene: Spatially Disentangled Generative Radiance Fields for Controllable 3D-aware Scene Synthesis"}
    assert same_work({"title": "DisCoScene: Spatially disentangled generative radiance fields for controllable "
                               "3D-aware scene synthesis"}, me)
    assert not same_work({"title": "Neural assets: 3d-aware multi-object scene synthesis"}, me)
