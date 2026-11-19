"""Persistent FAISS index with explicit vector-id -> chunk-id mapping in SQLite.

Vector ids are assigned by us (``IndexIDMap2``) and recorded in the
``vector_map`` table, so a FAISS hit can always be resolved back to its chunk
text in SQLite. Vectors are L2-normalised, so inner product == cosine.
"""
from __future__ import annotations

import re
from pathlib import Path

import faiss
import numpy as np

from config import CONFIG, path
from database.db_manager import DatabaseManager
from utils.logging_config import get_logger

log = get_logger("faiss")


def index_path_for(model_name: str) -> Path:
    if model_name == CONFIG["embedding_model"]:
        return path("faiss_index_path")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
    base = path("faiss_index_path")
    return base.with_name(f"{base.stem}__{slug}{base.suffix}")


class FaissManager:
    def __init__(self, dim: int, model_name: str, db: DatabaseManager,
                 index_type: str | None = None, index_file: Path | None = None) -> None:
        self.dim = dim
        self.model_name = model_name
        self.db = db
        self.index_type = index_type or CONFIG["faiss_index_type"]
        self.index_file = index_file or index_path_for(model_name)
        self.index: faiss.Index | None = None

    # ---------------------------------------------------------------- build
    def _new_index(self, train: np.ndarray | None = None) -> faiss.Index:
        if self.index_type == "flat_ip":
            base = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "flat_l2":
            base = faiss.IndexFlatL2(self.dim)
        elif self.index_type == "ivf_flat":
            if train is None or len(train) < CONFIG["faiss_ivf_nlist"] * 39:
                raise ValueError("ivf_flat needs >= 39*nlist training vectors; use flat_ip for small corpora")
            quant = faiss.IndexFlatIP(self.dim)
            base = faiss.IndexIVFFlat(quant, self.dim, CONFIG["faiss_ivf_nlist"], faiss.METRIC_INNER_PRODUCT)
            base.train(train)
        else:
            raise ValueError(f"Unknown faiss_index_type {self.index_type}")
        return faiss.IndexIDMap2(base)

    def load_or_create(self) -> "FaissManager":
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            if self.index.d != self.dim:
                raise ValueError(f"Index dim {self.index.d} != model dim {self.dim}; rebuild required")
            log.info("Loaded FAISS index %s (%d vectors)", self.index_file.name, self.index.ntotal)
        else:
            self.index = self._new_index()
        return self

    def save(self) -> None:
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_file.with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp))
        tmp.replace(self.index_file)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def next_id(self) -> int:
        row = self.db.conn.execute("SELECT MAX(vector_id) FROM vector_map WHERE embedding_model=?",
                                   (self.model_name,)).fetchone()
        return (row[0] + 1) if row[0] is not None else 0

    def add(self, vectors: np.ndarray, chunk_ids: list[str]) -> list[int]:
        """Add vectors and record their mapping. The index is saved by the caller."""
        assert vectors.shape == (len(chunk_ids), self.dim)
        start = self.next_id()
        ids = np.arange(start, start + len(chunk_ids), dtype="int64")
        self.index.add_with_ids(vectors, ids)
        self.db.add_vector_mappings(self.model_name, list(zip(ids.tolist(), chunk_ids)))
        return ids.tolist()

    def remove_stale(self) -> int:
        """Drop vectors whose chunk mapping no longer exists (e.g. a paper was re-chunked)."""
        if self.ntotal == 0:
            return 0
        in_index = set(faiss.vector_to_array(self.index.id_map).tolist())
        mapped = {r[0] for r in self.db.conn.execute(
            "SELECT vector_id FROM vector_map WHERE embedding_model=?", (self.model_name,))}
        stale = in_index - mapped
        if stale:
            self.index.remove_ids(np.array(sorted(stale), dtype="int64"))
        return len(stale)

    def reset(self) -> None:
        """Drop the index and its mappings (used when the model or chunking changes)."""
        with self.db.transaction() as c:
            c.execute("DELETE FROM vector_map WHERE embedding_model=?", (self.model_name,))
            if self.model_name == CONFIG["embedding_model"]:
                c.execute("UPDATE chunks SET embedding_status='pending', embedding_model=NULL")
        self.index = self._new_index()
        self.index_file.unlink(missing_ok=True)

    # --------------------------------------------------------------- search
    def search(self, queries: np.ndarray, k: int, exclude_paper_ids: set[str] | None = None) -> list[list[dict]]:
        """Return per query a ranked list of {chunk_id, vector_id, score, ...chunk metadata}."""
        if self.ntotal == 0:
            return [[] for _ in range(len(queries))]
        # Over-fetch when excluding papers so we still return k hits.
        fetch = min(self.ntotal, k * 3 if exclude_paper_ids else k)
        if isinstance(faiss.downcast_index(self.index.index), faiss.IndexIVF):
            faiss.downcast_index(self.index.index).nprobe = 10
        scores, ids = self.index.search(queries.astype("float32"), fetch)
        all_ids = sorted({int(i) for i in ids.ravel() if i >= 0})
        mapping = self.db.chunk_ids_for_vectors(self.model_name, all_ids)
        chunks = {c["chunk_id"]: c for c in self.db.get_chunks_by_ids(list(set(mapping.values())))}
        results = []
        for qscores, qids in zip(scores, ids):
            hits = []
            for s, vid in zip(qscores, qids):
                cid = mapping.get(int(vid))
                if cid is None or cid not in chunks:
                    continue
                ch = chunks[cid]
                if exclude_paper_ids and ch["paper_id"] in exclude_paper_ids:
                    continue
                score = float(s) if self.index_type != "flat_l2" else float(1 - s / 2)  # L2 on unit vectors -> cosine
                hits.append({**ch, "vector_id": int(vid), "similarity_score": score})
                if len(hits) == k:
                    break
            for rank, h in enumerate(hits, 1):
                h["initial_rank"] = rank
            results.append(hits)
        return results

    def reconstruct(self, vector_ids: list[int]) -> np.ndarray:
        return np.vstack([self.index.reconstruct(int(v)) for v in vector_ids]) if vector_ids else np.zeros((0, self.dim), "float32")

    # ----------------------------------------------------------- validation
    def validate(self) -> dict:
        """Check that FAISS and SQLite agree on the set of vectors."""
        n_map = self.db.conn.execute("SELECT COUNT(*) FROM vector_map WHERE embedding_model=?",
                                     (self.model_name,)).fetchone()[0]
        problems = []
        if n_map != self.ntotal:
            problems.append(f"vector_map has {n_map} rows but FAISS has {self.ntotal} vectors")
        if self.ntotal:
            sample = [r[0] for r in self.db.conn.execute(
                "SELECT vector_id FROM vector_map WHERE embedding_model=? ORDER BY RANDOM() LIMIT 20",
                (self.model_name,))]
            for vid in sample:
                try:
                    self.index.reconstruct(int(vid))
                except RuntimeError:
                    problems.append(f"vector {vid} missing from FAISS")
                    break
        dangling = self.db.conn.execute(
            "SELECT COUNT(*) FROM vector_map v LEFT JOIN chunks c ON c.chunk_id=v.chunk_id "
            "WHERE v.embedding_model=? AND c.chunk_id IS NULL", (self.model_name,)).fetchone()[0]
        if dangling:
            problems.append(f"{dangling} mappings point to missing chunks")
        return {"ok": not problems, "vectors": self.ntotal, "mappings": n_map, "problems": problems}
