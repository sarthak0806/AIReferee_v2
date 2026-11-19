import pytest

from retrieval.cross_encoder import get_reranker


@pytest.fixture(scope="module")
def rr():
    return get_reranker()


def test_relevant_passage_ranked_first(rr):
    q = "graph neural networks for traffic forecasting"
    cands = [{"chunk_id": "x", "chunk_text": "We analyse Arabic dialect syntax in tweets."},
             {"chunk_id": "y", "chunk_text": "We propose a spatio-temporal graph neural network that forecasts road "
                                             "traffic speed from sensor graphs."}]
    out = rr.rerank(q, cands)
    assert out[0]["chunk_id"] == "y" and out[0]["final_rank"] == 1
    assert out[0]["reranker_score"] > out[1]["reranker_score"]


def test_rerank_top_k_and_inputs_untouched(rr):
    cands = [{"chunk_id": str(i), "chunk_text": f"passage {i}"} for i in range(5)]
    out = rr.rerank("passage", cands, top_k=2)
    assert len(out) == 2 and "reranker_score" not in cands[0]


@pytest.mark.corpus
def test_bi_encoder_vs_cross_encoder_metrics_computed():
    """Reports both rankings; does not assume reranking helps (that is what the benchmark measures)."""
    from embeddings.benchmark import benchmark_rerank
    from config import CONFIG
    res = benchmark_rerank([CONFIG["reranker_model"]])
    for kind in ("title", "abstract"):
        assert "mrr" in res[kind]["bi_encoder_only"] and "mrr" in res[kind][CONFIG["reranker_model"]]
        # a reranker can only reorder the first stage: recall@30 bounds what it can recover
        assert res[kind][CONFIG["reranker_model"]]["recall@10"] <= res[kind]["first_stage_recall@30"] + 1e-9
