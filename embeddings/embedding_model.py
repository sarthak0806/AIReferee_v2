"""Sentence-Transformer bi-encoder wrapper with model-specific prompts."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import CONFIG
from utils.hf import load_cached
from utils.logging_config import get_logger

log = get_logger("embedding_model")

# Query/document prefixes recommended by each model's authors.
MODEL_SPECS: dict[str, dict] = {
    "sentence-transformers/all-MiniLM-L6-v2": {},
    "BAAI/bge-small-en-v1.5": {"query_prefix": "Represent this sentence for searching relevant passages: "},
    "BAAI/bge-base-en-v1.5": {"query_prefix": "Represent this sentence for searching relevant passages: "},
    "nomic-ai/nomic-embed-text-v1": {"query_prefix": "search_query: ", "doc_prefix": "search_document: ",
                                     "trust_remote_code": True},
}


class EmbeddingModel:
    def __init__(self, model_name: str | None = None, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name or CONFIG["embedding_model"]
        spec = MODEL_SPECS.get(self.name, {})
        self.query_prefix = spec.get("query_prefix", "")
        self.doc_prefix = spec.get("doc_prefix", "")
        self.model = load_cached(SentenceTransformer, self.name, device=device,
                                 trust_remote_code=spec.get("trust_remote_code", False))
        positions = getattr(self.model[0].auto_model.config, "max_position_embeddings", 512)
        self.model.max_seq_length = min(CONFIG["embedding_max_seq_length"], positions)
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = dim_fn()
        log.info("Loaded %s (dim=%d, max_seq_length=%d)", self.name, self.dim, self.model.max_seq_length)

    @property
    def version(self) -> str:
        """Model id plus the resolved weights revision when available."""
        rev = getattr(self.model, "model_card_data", None)
        base = getattr(rev, "base_model_revision", None) if rev else None
        return f"{self.name}@{base}" if base else self.name

    def _encode(self, texts: list[str], batch_size: int | None = None, progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size or CONFIG["embedding_batch_size"],
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=progress)
        return vecs.astype("float32")

    def encode_documents(self, texts: list[str], progress: bool = False) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts], progress=progress)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    return EmbeddingModel(model_name)
