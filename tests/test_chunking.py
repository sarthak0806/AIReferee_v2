import pytest

from config import CONFIG
from database.db_manager import DatabaseManager
from ingestion.chunker import chunk_document, get_tokenizer
from ingestion.extractor import ExtractedDocument, Paragraph, Section


def _doc(n_sentences=200):
    sents = " ".join(f"Sentence number {i} explains part {i % 7} of the method in detail." for i in range(n_sentences))
    return ExtractedDocument(
        title="T", abstract="A",
        sections=[Section("Abstract", "Abstract", 1, [Paragraph(1, "Short abstract sentence here. " * 15)]),
                  Section("Methodology", "3 Method", 2, [Paragraph(2, sents[: len(sents) // 2]),
                                                         Paragraph(3, sents[len(sents) // 2:])]),
                  Section("References", "References", 5, [Paragraph(5, "[1] Ref. " * 50)])],
        page_count=5, method="test", char_count=len(sents))


def ntok(text):
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def test_chunks_respect_size_and_sections():
    chunks = chunk_document(_doc(), "p1", "cvpr", chunk_size=120, overlap=30)
    assert chunks, "expected chunks"
    assert all(c.token_count <= 120 for c in chunks)
    assert {c.section_name for c in chunks} == {"Abstract", "Methodology"}   # References excluded
    assert all(c.paper_id == "p1" and c.conference == "cvpr" for c in chunks)
    # chunks never cross section boundaries
    assert not any("Short abstract" in c.chunk_text and "Sentence number" in c.chunk_text for c in chunks)


def test_overlap_between_consecutive_chunks():
    chunks = [c for c in chunk_document(_doc(), "p1", None, chunk_size=120, overlap=30)
              if c.section_name == "Methodology"]
    assert len(chunks) >= 3
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.chunk_text.split(". ")[-1].strip(". ")
        assert last_sentence in b.chunk_text          # whole sentences carried over


def test_sentences_are_not_split():
    for c in chunk_document(_doc(), "p1", None, chunk_size=120, overlap=30):
        if c.section_name == "Methodology":
            assert c.chunk_text.startswith("Sentence number") and c.chunk_text.endswith(".")


def test_ids_are_stable_and_sequential():
    a = chunk_document(_doc(), "p1", None, chunk_size=120, overlap=30)
    b = chunk_document(_doc(), "p1", None, chunk_size=120, overlap=30)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.chunk_index for c in a] == list(range(len(a)))


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_document(_doc(), "p1", None, chunk_size=50, overlap=50)


@pytest.mark.corpus
def test_corpus_chunks_within_limits():
    db = DatabaseManager()
    mx, mn, n = db.conn.execute("SELECT MAX(token_count), MIN(token_count), COUNT(*) FROM chunks").fetchone()
    assert n > 0 and mx <= CONFIG["chunk_size"] and mn >= CONFIG["min_chunk_tokens"]
    leaked = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE section_name IN ('References','Front Matter')"
                             ).fetchone()[0]
    assert leaked == 0
    boiler = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE chunk_text LIKE '%Open Access version%' "
                             "OR chunk_text LIKE '%Permission to make digital%'").fetchone()[0]
    assert boiler == 0
    db.close()
