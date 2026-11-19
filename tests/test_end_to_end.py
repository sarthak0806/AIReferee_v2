"""End-to-end pipeline test on a real held-out paper (not in the reference corpus).

Marked ``e2e``: needs the built corpus/index, network, and uses Groq + SerpAPI
quota when keys are configured. Run with:  python -m pytest -m e2e
"""
from pathlib import Path

import pytest

from config import CONFIG, PROJECT_ROOT
from evaluation.final_evaluator import csv_row, evaluate_paper
from reporting.csv_generator import to_csv_string, validate_row
from reporting.pdf_report import build_pdf
from utils.text_utils import word_count

PAPER = PROJECT_ROOT / "data" / "test_uploads" / "heldout_cvpr2023.pdf"


@pytest.fixture(scope="module")
def result():
    if not PAPER.exists():
        pytest.skip("held-out test paper not downloaded")
    return evaluate_paper(PAPER, paper_id="E2E_CVPR2023", use_external=True)


@pytest.mark.e2e
def test_profile_and_queries(result):
    assert result["analysis"]["profile"]["title"]
    assert 3 <= len(result["queries"]) <= 12
    assert all(q["query"] and q["category"] for q in result["queries"])


@pytest.mark.e2e
def test_retrieval_stages(result):
    assert all(len(h) <= CONFIG["top_k_similarity"] for h in result["internal_results"])
    assert any(result["internal_results"])
    ev = result["evidence"]
    assert 1 <= len(ev) <= CONFIG["final_evidence_count"]
    for e in ev:
        assert e["content_type"] == "full_text" and e["passage"] and e["title"]
        assert e["source_url"] or e["doi"]
    for x in result["external_results"]:
        assert x["content_type"] == "metadata_only"


@pytest.mark.e2e
def test_decision_and_outputs(result):
    d = result["decision"]
    assert d["decision"] in ("PUBLISHABLE", "NON-PUBLISHABLE", "HUMAN_REVIEW")
    row = csv_row(result)
    assert not validate_row(row)
    if row["Publishable"] == 1:
        assert row["Conference"] in CONFIG["conferences"] and row["Conference"] != "daa"
        assert word_count(row["Rationale"]) <= CONFIG["max_rationale_words"]
    else:
        assert row["Conference"] == "na" and row["Rationale"] == "na"
    assert to_csv_string([row]).startswith("Paper ID,Publishable,Conference,Rationale")
    pdf = build_pdf(result)
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000
    assert (PROJECT_ROOT / "data" / "reports" / "E2E_CVPR2023" / "result.json").exists()
