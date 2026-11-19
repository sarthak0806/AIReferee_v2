"""Collection tests.

Unit tests run offline. Corpus tests (marked ``corpus``) check the real
collected data in data/aireferee.db and data/reference_papers/.
"""
import hashlib

import pymupdf as fitz
import pytest

from config import CONFIG, CONFERENCE_SOURCES, PROJECT_ROOT, active_conferences, conference_dir, target_for
from database.db_manager import DatabaseManager
from ingestion.conference_collector import ArxivVenueAdapter
from ingestion.downloader import validate_pdf


# ------------------------------------------------------------------ unit tests
def _kdd_adapter():
    return ArxivVenueAdapter("kdd", CONFERENCE_SOURCES["kdd"][0])


@pytest.mark.parametrize("comment", ["Accepted at KDD 2024", "To appear in KDD'24", "accepted by SIGKDD 2024 research track"])
def test_arxiv_accepts_declared_acceptance(comment):
    assert _kdd_adapter().verify(comment, "")


@pytest.mark.parametrize("comment", ["Submitted to KDD 2024", "Under review at KDD 2024",
                                     "Accepted at the KDD 2024 workshop on X", "We compare with KDD 2024 methods", ""])
def test_arxiv_rejects_unverified(comment):
    assert _kdd_adapter().verify(comment, "") is None


def _make_pdf(pages: int, text: str) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page().insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_validate_pdf_rules():
    assert not validate_pdf(b"<html>challenge</html>").ok
    assert validate_pdf(_make_pdf(2, "x" * 100)).detail.startswith("only 2 pages")
    assert validate_pdf(_make_pdf(5, "")).status == "needs_ocr"


def test_target_follows_config(monkeypatch):
    monkeypatch.setitem(CONFIG, "papers_per_conference", 334)
    assert target_for("cvpr") == 334
    monkeypatch.setitem(CONFIG, "max_total_papers", 600)
    assert target_for("cvpr") == 100


# ---------------------------------------------------------------- corpus tests
@pytest.fixture(scope="module")
def db():
    d = DatabaseManager()
    yield d
    d.close()


CONFS_WITH_SOURCES = [c for c in active_conferences() if CONFERENCE_SOURCES.get(c)]


@pytest.mark.corpus
@pytest.mark.parametrize("conf", CONFS_WITH_SOURCES)
def test_target_reached(db, conf):
    assert db.count_valid(conf) >= target_for(conf)


@pytest.mark.corpus
@pytest.mark.parametrize("conf", CONFS_WITH_SOURCES)
def test_valid_papers_are_real_and_labelled(db, conf):
    for p in db.list_papers(conf, "valid"):
        assert p["conference"] == conf
        assert p["venue_verification"], p["paper_id"]
        f = PROJECT_ROOT / p["pdf_path"]
        assert f.parent == conference_dir(conf)
        data = f.read_bytes()
        assert hashlib.sha256(data).hexdigest() == p["file_sha256"]
        assert validate_pdf(data).ok


@pytest.mark.corpus
def test_no_duplicate_papers(db):
    valid = [p for c in CONFIG["conferences"] for p in db.list_papers(c, "valid")]
    titles = [p["title"].lower().strip() for p in valid]
    assert len(titles) == len(set(titles))
    assert len({p["file_sha256"] for p in valid}) == len(valid)


@pytest.mark.corpus
def test_no_untracked_pdfs_on_disk(db):
    tracked = {p["pdf_path"] for c in CONFIG["conferences"] for p in db.list_papers(c, "valid")}
    for conf in CONFIG["conferences"]:
        for f in conference_dir(conf).glob("*.pdf"):
            assert f.relative_to(PROJECT_ROOT).as_posix() in tracked
