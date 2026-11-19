"""LLM-based multi-query generation (Groq) with validation and de-duplication."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz

from config import CONFIG
from evaluation.paper_analyzer import ResearchProfile
from utils.llm import LLMUnavailable, complete_json, llm_available, model_tag
from utils.logging_config import get_logger

log = get_logger("query_generator")

Category = Literal["research_problem", "methodology", "novelty", "dataset_experiments", "results", "limitations"]


class ScholarlyQuery(BaseModel):
    query: str = Field(min_length=8, max_length=300)
    category: Category
    reason: str = ""

    @field_validator("query")
    @classmethod
    def strip(cls, v: str) -> str:
        return " ".join(v.split())


class QuerySet(BaseModel):
    queries: list[ScholarlyQuery] = Field(min_length=1, max_length=12)


PROMPT = """Generate {n} distinct scholarly search queries (5-15 words each, keyword style, no quotes or boolean
operators) to find prior work for reviewing the paper profile below. Cover these categories, at least one each
where the profile has information:
- research_problem: existing research on the same problem
- methodology: papers using similar methods, algorithms or architectures
- novelty: the closest prior work to the claimed contribution
- dataset_experiments: studies using the same datasets, benchmarks or metrics
- results: comparable experimental results
- limitations: research addressing similar gaps or limitations
Do not repeat the paper's title verbatim. Each query needs a one-sentence reason.

<paper>
{profile}
</paper>"""


def _dedupe(queries: list[ScholarlyQuery], threshold: int = 85) -> list[ScholarlyQuery]:
    out: list[ScholarlyQuery] = []
    for q in queries:
        if all(fuzz.token_set_ratio(q.query.lower(), o.query.lower()) < threshold for o in out):
            out.append(q)
    return out


def fallback_queries(p: ResearchProfile) -> list[ScholarlyQuery]:
    """Deterministic templates from profile fields when the LLM is unavailable."""
    def clip(s: str, n: int = 14) -> str:
        return " ".join((s or "").split()[:n])

    cands = [
        (clip(p.research_problem or p.abstract), "research_problem", "Same research problem."),
        (clip(p.methodology), "methodology", "Similar methods."),
        (clip(p.claimed_contributions[0] if p.claimed_contributions else p.title), "novelty", "Closest prior work."),
        (clip(" ".join(p.datasets[:3] + p.metrics[:2]) + " " + clip(p.title, 6)), "dataset_experiments", "Same datasets."),
        (clip(p.results), "results", "Comparable results."),
        (clip(p.limitations[0] if p.limitations else p.conclusion), "limitations", "Similar gaps."),
    ]
    out = []
    for q, cat, why in cands:
        if len(q) >= 8:
            out.append(ScholarlyQuery(query=q, category=cat, reason=why))
    return _dedupe(out)


def generate_queries(profile: ResearchProfile, n: int | None = None) -> tuple[list[ScholarlyQuery], str]:
    n = n or CONFIG["num_queries"]
    if llm_available():
        try:
            fields = profile.model_dump(exclude={"abstract"})
            text = f"title: {profile.title}\nabstract: {profile.abstract[:1500]}\n" + "\n".join(
                f"{k}: {v}" for k, v in fields.items() if v and k != "title")
            qs = complete_json(PROMPT.format(n=n, profile=text[:8000]), QuerySet, max_tokens=2000)
            return _dedupe(qs.queries)[:n], model_tag()
        except LLMUnavailable as exc:
            log.warning("Query generation fell back to templates: %s", exc)
    return fallback_queries(profile)[:n], "heuristic-fallback"
