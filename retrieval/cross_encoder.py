"""Stage-2 Cross-Encoder reranking of (query, passage) pairs."""
from __future__ import annotations

from functools import lru_cache

from config import CONFIG
from utils.hf import load_cached
from utils.logging_config import get_logger

log = get_logger("cross_encoder")


class Reranker:
    def __init__(self, model_name: str | None = None, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name or CONFIG["reranker_model"]
        self.model = load_cached(CrossEncoder, self.name, device=device, max_length=512)
        log.info("Loaded reranker %s", self.name)

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        if not pairs:
            return []
        # Raw logits (no sigmoid) so thresholds in config are comparable across calls.
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False,
                                    activation_fn=None, convert_to_numpy=True)
        return [float(s) for s in scores]

    def rerank(self, query: str, candidates: list[dict], text_key: str = "chunk_text",
               top_k: int | None = None) -> list[dict]:
        """Return candidates sorted by reranker score (copies with ``reranker_score`` / ``final_rank``)."""
        top_k = top_k or len(candidates)
        scores = self.score([(query, c.get(text_key) or "") for c in candidates])
        ranked = sorted(({**c, "reranker_score": s} for c, s in zip(candidates, scores)),
                        key=lambda c: c["reranker_score"], reverse=True)[:top_k]
        for i, c in enumerate(ranked, 1):
            c["final_rank"] = i
        return ranked


@lru_cache(maxsize=2)
def get_reranker(model_name: str | None = None) -> Reranker:
    return Reranker(model_name)
