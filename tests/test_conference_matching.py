import numpy as np
import pytest

from config import CONFIG
from evaluation.conference_matcher import keyword_score, match, neighbour_votes
from evaluation.final_evaluator import csv_row
from evaluation.rationale_generator import conference_rationale
from reporting.csv_generator import to_csv_string, validate_row
from utils.text_utils import truncate_words, word_count


def _profiles():
    return {"cvpr": {"keywords": {"image": 0.5, "segmentation": 0.3}, "llm_profile": {}},
            "emnlp": {"keywords": {"language": 0.5, "translation": 0.3}, "llm_profile": {}}}


def test_match_prefers_nearest_venue(monkeypatch):
    monkeypatch.setitem(CONFIG, "enable_llm", False)
    cents = {"cvpr": np.array([1.0, 0.0]), "emnlp": np.array([0.0, 1.0])}
    hits = [[{"conference": "cvpr", "paper_id": "c1", "similarity_score": 0.8, "chunk_text": "x", "title": "C1"},
             {"conference": "emnlp", "paper_id": "e1", "similarity_score": 0.2, "chunk_text": "y", "title": "E1"}]]
    m = match(np.array([0.9, 0.1]), "image segmentation results", "profile", hits, _profiles(), cents)
    assert m["recommended"] == "cvpr"
    assert m["ranking"][0]["top_reference_papers"][0]["paper_id"] == "c1"
    assert "llm" not in m["weights_used"]


def test_keyword_and_neighbour_signals():
    assert keyword_score("Image segmentation of images", {"image": 0.5, "language": 0.5}) == 0.5
    shares, top = neighbour_votes([[{"conference": "kdd", "paper_id": "k", "similarity_score": 0.6, "chunk_text": "t"},
                                    {"conference": "cvpr", "paper_id": "c", "similarity_score": 0.2, "chunk_text": "t"}]])
    assert shares["kdd"] == pytest.approx(0.75)


def test_truncate_words_limit():
    text = "word " * 150 + "end."
    out = truncate_words(text, 100)
    assert word_count(out) <= 100


def test_template_rationale_within_limit(monkeypatch):
    monkeypatch.setitem(CONFIG, "enable_llm", False)
    text, src = conference_rationale("cvpr", "profile", {"keywords": {"image": 1, "video": 1}, "llm_profile": {}},
                                     {"top_reference_papers": [{"title": "A CVPR Paper", "similarity_score": 0.7,
                                                                "passage": "x"}]})
    assert src == "template-fallback" and word_count(text) <= CONFIG["max_rationale_words"]
    assert "A CVPR Paper" in text


def test_csv_rules():
    pub = {"paper_id": "P001", "decision": {"csv_publishable": 1},
           "conference": {"recommended": "kdd", "rationale": "Fits KDD."}}
    non = {"paper_id": "P002", "decision": {"csv_publishable": 0}, "conference": None}
    rows = [csv_row(pub), csv_row(non)]
    assert rows[1] == {"Paper ID": "P002", "Publishable": 0, "Conference": "na", "Rationale": "na"}
    csv = to_csv_string(rows)
    assert csv.splitlines()[0] == "Paper ID,Publishable,Conference,Rationale"
    assert csv.splitlines()[2] == "P002,0,na,na"
    assert validate_row({"Paper ID": "x", "Publishable": 1, "Conference": "na", "Rationale": "r"})
    assert validate_row({"Paper ID": "x", "Publishable": 1, "Conference": "kdd", "Rationale": "w " * 101})
    assert validate_row({"Paper ID": "x", "Publishable": 0, "Conference": "kdd", "Rationale": "na"})


@pytest.mark.corpus
def test_leave_one_out_conference_accuracy_reported():
    from embeddings.benchmark import benchmark_conference
    res = benchmark_conference()
    assert res["papers"] >= 100 and 0 <= res["combined"]["accuracy"] <= 1
    # better than the 20% chance level for 5 venues
    assert res["combined"]["accuracy"] > 0.2
