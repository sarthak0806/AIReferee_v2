"""Shared rubric types for the six evaluation dimensions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field, field_validator

DIMENSIONS = ["novelty", "methodology", "experimental", "literature", "results", "reproducibility"]


@dataclass
class DimensionSpec:
    name: str
    label: str
    questions: list[str]
    signals: Callable[..., dict]   # (doc, analyzed, evidence, external) -> dict of objective text signals
    fallback_score: Callable[[dict], float]


class DimensionAssessment(BaseModel):
    score: float = Field(ge=0, le=10)
    finding: str = Field(description="One-paragraph finding for this dimension")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list, description="Evidence ids like E1, X2")
    counter_evidence: list[str] = Field(default_factory=list, description="Evidence ids that contradict claims")
    missing_evidence: list[str] = Field(default_factory=list, description="What the paper should show but does not")
    confidence: float = Field(ge=0, le=1)
    critical_flaw: bool = False
    critical_flaw_reason: str = ""

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(float(v), 1)


class EvaluationOutput(BaseModel):
    novelty: DimensionAssessment
    methodology: DimensionAssessment
    experimental: DimensionAssessment
    literature: DimensionAssessment
    results: DimensionAssessment
    reproducibility: DimensionAssessment
    coherence_issues: list[str] = Field(default_factory=list,
                                        description="Incoherent arguments or poorly organised content")
    unsubstantiated_claims: list[str] = Field(default_factory=list)
    overall_summary: str = ""
    revision_roadmap: list[str] = Field(default_factory=list)


def clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))
