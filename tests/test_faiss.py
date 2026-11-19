import numpy as np
import pytest

from config import CONFIG
from database.db_manager import DatabaseManager
from retrieval.faiss_manager import FaissManager

DIM = 16


@pytest.fixture
def setup(tmp_path):
    db = DatabaseManager(tmp_path / "t.db")
    for p in range(3):
        db.insert_paper({"paper_id": f"p{p}", "title": f"Paper {p}", "conference": "cvpr",
                         "source_url": f"https://x/{p}", "source_name": "test"})
        db.insert_chunks([{"chunk_id": f"p{p}::{i}", "paper_id": f"p{p}", "conference": "cvpr",
                           "chunk_index": i, "chunk_text": f"text {p} {i}"} for i in range(4)])
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(12, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"p{p}::{i}" for p in range(3) for i in range(4)]
    fm = FaissManager(DIM, "test-model", db, index_type="flat_ip", index_file=tmp_path / "t.index").load_or_create()
    yield db, fm, vecs, ids, tmp_path
    db.close()


def test_insert_and_mapping(setup):
    db, fm, vecs, ids, _ = setup
    vids = fm.add(vecs, ids)
    assert vids == list(range(12)) and fm.ntotal == 12
    assert db.chunk_ids_for_vectors("test-model", [0, 11]) == {0: "p0::0", 11: "p2::3"}


def test_similarity_ordering_and_text_roundtrip(setup):
    db, fm, vecs, ids, _ = setup
    fm.add(vecs, ids)
    hits = fm.search(vecs[5:6], k=3)[0]
    assert hits[0]["chunk_id"] == ids[5] and hits[0]["chunk_text"] == "text 1 1"
    assert hits[0]["similarity_score"] == pytest.approx(1.0, abs=1e-4)
    assert [h["similarity_score"] for h in hits] == sorted([h["similarity_score"] for h in hits], reverse=True)
    assert [h["initial_rank"] for h in hits] == [1, 2, 3]


def test_exclude_papers(setup):
    db, fm, vecs, ids, _ = setup
    fm.add(vecs, ids)
    hits = fm.search(vecs[0:1], k=5, exclude_paper_ids={"p0"})[0]
    assert hits and all(h["paper_id"] != "p0" for h in hits) and len(hits) == 5


def test_persistence_and_incremental_ids(setup):
    db, fm, vecs, ids, tmp = setup
    fm.add(vecs[:6], ids[:6])
    fm.save()
    fm2 = FaissManager(DIM, "test-model", db, index_type="flat_ip", index_file=tmp / "t.index").load_or_create()
    assert fm2.ntotal == 6
    assert fm2.add(vecs[6:], ids[6:]) == list(range(6, 12))
    assert fm2.validate()["ok"]


def test_validate_detects_mismatch(setup):
    db, fm, vecs, ids, tmp = setup
    fm.add(vecs, ids)
    fm.save()
    db.delete_chunks("p1")              # cascades vector_map rows away
    assert not fm.validate()["ok"]


def test_duplicate_chunk_mapping_rejected(setup):
    import sqlite3
    db, fm, vecs, ids, _ = setup
    fm.add(vecs[:1], ids[:1])
    with pytest.raises(sqlite3.IntegrityError):
        fm.add(vecs[:1], ids[:1])


@pytest.mark.corpus
def test_corpus_index_consistent():
    from embeddings.embedding_generator import open_index
    db = DatabaseManager()
    fm, model = open_index(db)
    v = fm.validate()
    assert v["ok"], v["problems"]
    n_chunks = db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert v["vectors"] == n_chunks
    assert fm.index_file.exists() and model.name == CONFIG["embedding_model"]
    db.close()


def test_remove_stale_after_rechunk(setup):
    db, fm, vecs, ids, _ = setup
    fm.add(vecs, ids)
    db.delete_chunks("p1")                    # 4 mappings cascade away
    assert fm.remove_stale() == 4
    assert fm.validate()["ok"] and fm.ntotal == 8
