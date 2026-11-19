import sqlite3

import pytest

from database.db_manager import DatabaseManager


@pytest.fixture
def db(tmp_path):
    d = DatabaseManager(tmp_path / "t.db")
    yield d
    d.close()


def _paper(pid="cvpr_2024_x", url="https://example.org/x"):
    return {"paper_id": pid, "title": "T", "conference": "cvpr", "source_url": url,
            "source_name": "test", "authors": ["A B", "C D"]}


def test_insert_and_get_paper(db):
    assert db.insert_paper(_paper())
    p = db.get_paper("cvpr_2024_x")
    assert p["authors"] == ["A B", "C D"]
    assert p["download_status"] == "pending"


def test_duplicate_paper_prevented(db):
    assert db.insert_paper(_paper())
    assert not db.insert_paper(_paper())                       # same id
    assert not db.insert_paper(_paper(pid="other"))            # same source url
    assert len(db.list_papers()) == 1


def test_update_and_count_valid(db):
    db.insert_paper(_paper())
    db.update_paper("cvpr_2024_x", download_status="valid", file_sha256="abc")
    assert db.count_valid("cvpr") == 1
    assert db.find_by_sha("abc")["paper_id"] == "cvpr_2024_x"


def test_unknown_column_rejected(db):
    with pytest.raises(ValueError):
        db.insert_paper({**_paper(), "bogus": 1})


def test_chunks_and_foreign_keys(db):
    db.insert_paper(_paper())
    chunk = {"chunk_id": "cvpr_2024_x::0", "paper_id": "cvpr_2024_x", "conference": "cvpr",
             "section_name": "Intro", "page_number": 1, "chunk_index": 0, "chunk_text": "hello", "token_count": 1}
    assert db.insert_chunks([chunk]) == 1
    assert db.insert_chunks([chunk]) == 0                       # idempotent
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_chunks([{**chunk, "chunk_id": "z::0", "paper_id": "missing"}])


def test_vector_mapping_roundtrip(db):
    db.insert_paper(_paper())
    db.insert_chunks([{"chunk_id": "c0", "paper_id": "cvpr_2024_x", "chunk_index": 0, "chunk_text": "t"}])
    db.add_vector_mappings("m", [(7, "c0")])
    assert db.chunk_ids_for_vectors("m", [7]) == {7: "c0"}
    rows = db.get_chunks_by_ids(["c0"])
    assert rows[0]["chunk_text"] == "t" and rows[0]["title"] == "T"
    assert db.get_chunks(embedding_status="embedded")[0]["embedding_model"] == "m"
