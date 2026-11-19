"""Stage-1 bi-encoder retrieval over the persistent FAISS index."""
from __future__ import annotations

from config import CONFIG
from database.db_manager import DatabaseManager
from embeddings.embedding_model import get_embedding_model
from retrieval.faiss_manager import FaissManager


class SimilaritySearcher:
    def __init__(self, db: DatabaseManager, model_name: str | None = None) -> None:
        self.db = db
        self.model = get_embedding_model(model_name)
        self.fm = FaissManager(self.model.dim, self.model.name, db).load_or_create()

    def search(self, queries: list[str], k: int | None = None,
               exclude_paper_ids: set[str] | None = None) -> list[list[dict]]:
        if not queries:
            return []
        vecs = self.model.encode_queries(queries)
        return self.fm.search(vecs, k or CONFIG["top_k_similarity"], exclude_paper_ids)

    def search_vectors(self, vecs, k: int, exclude_paper_ids: set[str] | None = None) -> list[list[dict]]:
        return self.fm.search(vecs, k, exclude_paper_ids)
