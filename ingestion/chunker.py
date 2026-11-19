"""Section-aware, sentence-preserving chunking.

Chunks never cross a section boundary. Sentences are packed up to
``chunk_size`` tokens (counted with the embedding model's tokenizer) and the
last sentences of a chunk, worth about ``chunk_overlap`` tokens, are repeated at
the start of the next one. A single sentence longer than ``chunk_size`` is
split on token boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from config import CONFIG
from ingestion.extractor import ExtractedDocument
from utils.text_utils import split_sentences


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str | None = None):
    from transformers import AutoTokenizer

    from utils.hf import load_cached
    tok = load_cached(AutoTokenizer.from_pretrained, model_name or CONFIG["embedding_model"])
    # Used only to *count* tokens (long sentences are split afterwards), so lift the
    # model limit to avoid "sequence length is longer than ..." warnings.
    tok.model_max_length = 10**9
    return tok


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    conference: str | None
    section_name: str
    page_number: int
    chunk_index: int
    chunk_text: str
    token_count: int

    def as_row(self) -> dict:
        return self.__dict__.copy()


def chunk_signature() -> str:
    """Identifies the chunking configuration; stored so a change forces re-chunking."""
    return f"{CONFIG['embedding_model']}|{CONFIG['chunk_size']}|{CONFIG['chunk_overlap']}|{CONFIG['min_chunk_tokens']}"


def chunk_document(doc: ExtractedDocument, paper_id: str, conference: str | None,
                   chunk_size: int | None = None, overlap: int | None = None,
                   exclude_sections: list[str] | None = None) -> list[Chunk]:
    size = chunk_size or CONFIG["chunk_size"]
    overlap = CONFIG["chunk_overlap"] if overlap is None else overlap
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    exclude = set(CONFIG["chunk_exclude_sections"] if exclude_sections is None else exclude_sections)
    tok = get_tokenizer()

    def ntok(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    chunks: list[Chunk] = []

    def emit(sents: list[tuple[str, int, int]], section: str) -> None:
        text = " ".join(s for s, _, _ in sents).strip()
        n = sum(t for _, _, t in sents)
        if not text or n < CONFIG["min_chunk_tokens"]:
            return  # captions, stray headings and other fragments carry no retrievable content
        idx = len(chunks)
        chunks.append(Chunk(f"{paper_id}::{idx:04d}", paper_id, conference, section,
                            sents[0][1], idx, text, n))

    for section in doc.sections:
        if section.name in exclude:
            continue
        # (sentence, page, tokens)
        units: list[tuple[str, int, int]] = []
        for para in section.paragraphs:
            for sent in split_sentences(para.text):
                n = ntok(sent)
                if n <= size:
                    units.append((sent, para.page, n))
                    continue
                ids = tok.encode(sent, add_special_tokens=False)
                for start in range(0, len(ids), size - overlap):
                    piece = tok.decode(ids[start:start + size])
                    units.append((piece, para.page, len(ids[start:start + size])))
        window: list[tuple[str, int, int]] = []
        total = fresh = 0  # fresh = sentences not yet emitted in any chunk
        for unit in units:
            if window and total + unit[2] > size:
                emit(window, section.name)
                # carry the tail of the window as overlap
                carry, carried = [], 0
                for u in reversed(window):
                    if carried + u[2] > overlap:
                        break
                    carry.insert(0, u)
                    carried += u[2]
                if carried + unit[2] > size:  # overlap + long sentence would exceed chunk_size
                    carry, carried = [], 0
                window, total, fresh = carry, carried, 0
            window.append(unit)
            total += unit[2]
            fresh += 1
        if window and fresh:
            emit(window, section.name)
    return chunks
