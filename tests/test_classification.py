import pytest

from config import CONFIG
from evaluation.metrics import classification_metrics
from evaluation.publishability_classifier import (HUMAN_REVIEW, NON_PUBLISHABLE, PUBLISHABLE, _sanitize_refs,
                                                  decide, fallback_assessment)
from evaluation.rubric import DIMENSIONS, DimensionAssessment, EvaluationOutput


def make(scores: dict | float, conf=0.8, flaw: str | None = None) -> EvaluationOutput:
    if not isinstance(scores, dict):
        scores = {d: scores for d in DIMENSIONS}
    dims = {d: DimensionAssessment(score=scores[d], finding="f", confidence=conf,
                                   critical_flaw=(d == flaw), critical_flaw_reason="broken" if d == flaw else "")
            for d in DIMENSIONS}
    return EvaluationOutput(**dims)


def test_clear_publishable_and_non_publishable():
    assert decide(make(8.0), "t").decision == PUBLISHABLE
    assert decide(make(3.5), "t").decision == NON_PUBLISHABLE


def test_weighted_score_uses_config_weights():
    s = {d: 0.0 for d in DIMENSIONS} | {"novelty": 10.0}
    d = decide(make(s), "t")
    assert d.weighted_score == pytest.approx(10 * CONFIG["dimension_weights"]["novelty"])


def test_critical_flaw_overrides_high_scores():
    d = decide(make(9.0, flaw="methodology"), "t")
    assert d.decision == NON_PUBLISHABLE and d.blocking_issues and d.csv_publishable == 0


def test_critical_minimum_score_rule():
    s = {d: 9.0 for d in DIMENSIONS} | {"experimental": 2.0}
    assert decide(make(s), "t").decision == NON_PUBLISHABLE


def test_borderline_and_low_confidence_go_to_review(monkeypatch):
    t = CONFIG["publishable_threshold"]
    assert decide(make(t + 0.1), "t").decision == HUMAN_REVIEW
    assert decide(make(8.0, conf=0.2), "t").decision == HUMAN_REVIEW
    assert decide(make(8.0), "t", insufficient_input=True).decision == HUMAN_REVIEW


def test_human_review_csv_policy(monkeypatch):
    t = CONFIG["publishable_threshold"]
    monkeypatch.setitem(CONFIG, "human_review_csv_policy", "score_threshold")
    assert decide(make(t + 0.1), "t").csv_publishable == 1
    monkeypatch.setitem(CONFIG, "human_review_csv_policy", "non_publishable")
    assert decide(make(t + 0.1), "t").csv_publishable == 0


def test_invented_evidence_ids_removed():
    out = make(7.0)
    out.novelty.supporting_evidence = ["E1", "E99", "X2"]
    out = _sanitize_refs(out, {"E1", "X2"})
    assert out.novelty.supporting_evidence == ["E1", "X2"]


def test_fallback_is_low_confidence():
    sig = {"novelty": {"claimed_contributions": 2, "max_internal_similarity": 0.5, "highly_related_external_works": 0},
           "methodology": {"method_section_words": 1200, "has_formalism": True, "states_assumptions": True,
                           "baselines_named": 3},
           "experimental": {"experiment_section_words": 1500, "datasets": ["X"], "baselines": ["Y"],
                            "mentions_ablation": True, "reports_variance_or_significance": True},
           "literature": {"reference_count": 40, "has_related_work_section": True, "relevant_retrieved_works": 4,
                          "relevant_works_not_cited": ["a"]},
           "results": {"extreme_claims": [], "acknowledges_limitations": True, "has_conclusion": True},
           "reproducibility": {"code_link": True, "hyperparameters": True, "implementation_details": True,
                               "seeds_or_hardware": False}}
    out = fallback_assessment(sig)
    assert all(0 <= getattr(out, d).score <= 10 for d in DIMENSIONS)
    assert decide(out, "heuristic-fallback").decision == HUMAN_REVIEW   # confidence 0.3 < min_confidence


def test_classification_metrics():
    m = classification_metrics([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert m["confusion_matrix"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    assert m["accuracy"] == 0.6 and m["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert m["recall"] == pytest.approx(0.6667, abs=1e-4) and m["f1"] == pytest.approx(0.6667, abs=1e-4)


def test_groq_retry_after_parsing():
    from utils.llm import _retry_after
    assert _retry_after("Please try again in 1.8825s. Need more") == pytest.approx(1.8825)
    assert _retry_after("Please try again in 1m2.5s.") == pytest.approx(62.5)
    assert _retry_after("other error") is None


def test_fit_to_budget_trims_paper(monkeypatch):
    from utils.llm import est_tokens, fit_to_budget
    monkeypatch.setitem(CONFIG, "llm_request_token_budget", 3000)
    prompt = "Assess.\n<paper>\n" + "word " * 8000 + "\n</paper>\nend"
    out, mt = fit_to_budget(prompt, "system", 2000)
    assert mt >= 1500 and est_tokens(out) + est_tokens("system") + mt <= 3000 + 100
    assert "<paper>" in out and "</paper>" in out and "truncated" in out


def test_model_chain_skips_exhausted_models(monkeypatch):
    import time
    from utils import llm
    monkeypatch.setitem(CONFIG, "llm_model", "m1")
    monkeypatch.setitem(CONFIG, "llm_fallback_models", ["m2", "m3"])
    monkeypatch.setattr(llm, "_blocked", {"m1": time.time() + 600})
    assert llm.model_chain() == ["m2", "m3"]
    assert "m1 resets in" in llm.quota_status()
    monkeypatch.setattr(llm, "_blocked", {m: time.time() + 600 for m in ("m1", "m2", "m3")})
    assert llm.model_chain() == []
