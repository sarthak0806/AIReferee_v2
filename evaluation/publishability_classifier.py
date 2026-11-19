"""Publishability classification.

The LLM assesses each dimension against the rubric, the retrieved evidence and
objective text signals. The decision itself is computed in code:
weighted score -> critical-failure rules -> confidence / borderline checks.
Weights and thresholds are configurable starting assumptions, not validated
publication standards.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from config import CONFIG
from evaluation import experiment_evaluator, literature_evaluator, methodology_evaluator, novelty_evaluator
from evaluation.rubric import DIMENSIONS, DimensionAssessment, EvaluationOutput
from utils.llm import LLMUnavailable, complete_json, llm_available, model_tag
from utils.logging_config import get_logger

log = get_logger("classifier")

SPECS = {
    "novelty": novelty_evaluator.SPEC,
    "methodology": methodology_evaluator.SPEC,
    "experimental": experiment_evaluator.EXPERIMENTAL,
    "literature": literature_evaluator.SPEC,
    "results": experiment_evaluator.RESULTS,
    "reproducibility": experiment_evaluator.REPRODUCIBILITY,
}

PUBLISHABLE, NON_PUBLISHABLE, HUMAN_REVIEW = "PUBLISHABLE", "NON-PUBLISHABLE", "HUMAN_REVIEW"


class Decision(BaseModel):
    decision: str
    weighted_score: float
    confidence: float
    blocking_issues: list[str]
    reasons: list[str]
    csv_publishable: int           # binary label after applying the submission policy
    assessment_source: str


COMMON = """You are reviewing a research paper for publishability at a top venue. The hackathon task flags papers
with inappropriate methodologies, incoherent arguments, poorly organised content, unsubstantiated claims,
unrealistic results, insufficient validation and weak contributions. Score each dimension 0-10
(0-3 fundamentally flawed, 4-5 major problems, 6-7 acceptable with revisions, 8-10 strong).
Rules: ground every judgement in the excerpt, signals and evidence. E* items are full-text passages from published
papers, X* items are metadata-only search results; cite them only by id. Similarity is not proof of plagiarism or
lack of novelty. critical_flaw=true only for a fundamental problem that alone makes the paper unpublishable.
Lower confidence when material is insufficient. Keep each text field under 60 words and lists under 5 items."""

PROMPT_A = COMMON + """

Assess: {questions}
coherence_issues: incoherent arguments or poorly organised content, if any.

<paper>
{paper}
</paper>

Objective signals: {signals}

<evidence>
{evidence}
</evidence>"""

PROMPT_B = COMMON + """

Assess: {questions}
Also give unsubstantiated_claims, an overall_summary (<= 80 words) covering ALL six dimensions, and a
revision_roadmap of concrete experiments, comparisons or clarifications.
Findings already made on the other dimensions: {prior}

<paper>
{paper}
</paper>

Objective signals: {signals}"""


class GroupA(BaseModel):
    novelty: DimensionAssessment
    methodology: DimensionAssessment
    literature: DimensionAssessment
    coherence_issues: list[str] = []


class GroupB(BaseModel):
    experimental: DimensionAssessment
    results: DimensionAssessment
    reproducibility: DimensionAssessment
    unsubstantiated_claims: list[str] = []
    overall_summary: str = ""
    revision_roadmap: list[str] = []


def format_evidence(evidence: list[dict], external: list[dict], max_chars: int = 450) -> str:
    lines = []
    for e in evidence:
        lines.append(f"[{e['evidence_id']}] {e['title']} ({(e.get('conference') or '').upper()} {e.get('year') or ''}), "
                     f"section {e.get('section')}, p.{e.get('page')}, relevance={e['relevance']}\n"
                     f"{e['passage'][:max_chars]}")
    for x in external:
        body = (x.get("abstract") or x.get("snippet") or "")[:250]
        lines.append(f"[{x['evidence_id']}] (metadata only, {x.get('sources')}) {x['title']} ({x.get('year') or 'n.d.'})\n{body}")
    return "\n\n".join(lines) or "(no evidence retrieved)"


def compute_signals(doc, analyzed, evidence, external) -> dict:
    return {name: spec.signals(doc, analyzed, evidence, external) for name, spec in SPECS.items()}


def fallback_assessment(signals: dict) -> EvaluationOutput:
    dims = {}
    for name, spec in SPECS.items():
        s = signals[name]
        dims[name] = DimensionAssessment(
            score=spec.fallback_score(s),
            finding=f"Heuristic estimate from text signals only (no LLM): {json.dumps(s, default=str)[:400]}",
            confidence=0.3, missing_evidence=["LLM assessment unavailable; human review recommended"],
        )
    return EvaluationOutput(**dims, overall_summary="Heuristic-only assessment (LLM unavailable).")


def _sanitize_refs(out: EvaluationOutput, valid_ids: set[str]) -> EvaluationOutput:
    """Drop evidence ids the model invented."""
    for name in DIMENSIONS:
        d: DimensionAssessment = getattr(out, name)
        d.supporting_evidence = [i for i in d.supporting_evidence if i in valid_ids]
        d.counter_evidence = [i for i in d.counter_evidence if i in valid_ids]
    return out


def _questions(names: list[str]) -> str:
    return " ".join(f"[{n}] " + " ".join(SPECS[n].questions) for n in names)


def _excerpt(doc, names: list[str], chars: int) -> str:
    parts = [f"TITLE: {doc.title}", f"ABSTRACT: {doc.abstract[:1500]}"]
    per = chars // max(1, len(names))
    for n in names:
        t = doc.section_text(n)
        if t:
            parts.append(f"## {n.upper()}\n{t[:per]}")
    return "\n".join(parts)


def assess(doc, signals: dict, evidence: list[dict], external: list[dict]) -> tuple[EvaluationOutput, str]:
    """Two focused LLM calls (each sized to the per-request token budget), then merged."""
    if llm_available():
        sig = lambda names: json.dumps({n: signals[n] for n in names}, default=str, separators=(",", ":"))[:1500]
        a_dims, b_dims = ["novelty", "methodology", "literature"], ["experimental", "results", "reproducibility"]
        try:
            a = complete_json(PROMPT_A.format(
                questions=_questions(a_dims), signals=sig(a_dims), evidence=format_evidence(evidence, external),
                paper=_excerpt(doc, ["Introduction", "Related Work", "Methodology"], 6000)),
                GroupA, max_tokens=3200, reasoning="medium")
            prior = "; ".join(f"{n}: {getattr(a, n).score}/10 - {getattr(a, n).finding[:200]}" for n in a_dims)
            b = complete_json(PROMPT_B.format(
                questions=_questions(b_dims), signals=sig(b_dims), prior=prior,
                paper=_excerpt(doc, ["Methodology", "Experiments", "Results", "Discussion", "Conclusion",
                                     "Limitations"], 8500)),
                GroupB, max_tokens=3200, reasoning="medium")
            out = EvaluationOutput(**{n: getattr(a, n) for n in a_dims}, **{n: getattr(b, n) for n in b_dims},
                                   coherence_issues=a.coherence_issues, unsubstantiated_claims=b.unsubstantiated_claims,
                                   overall_summary=b.overall_summary, revision_roadmap=b.revision_roadmap)
            ids = {e["evidence_id"] for e in evidence} | {x["evidence_id"] for x in external}
            return _sanitize_refs(out, ids), model_tag()
        except LLMUnavailable as exc:
            log.warning("Assessment fell back to heuristics: %s", exc)
    return fallback_assessment(signals), "heuristic-fallback"


def decide(out: EvaluationOutput, source: str, insufficient_input: bool = False) -> Decision:
    weights = CONFIG["dimension_weights"]
    scores = {d: getattr(out, d).score for d in DIMENSIONS}
    weighted = round(sum(weights[d] * scores[d] for d in DIMENSIONS) / sum(weights.values()), 2)
    confidence = round(sum(getattr(out, d).confidence for d in DIMENSIONS) / len(DIMENSIONS), 2)

    blocking = []
    for d, minimum in CONFIG["critical_min_scores"].items():
        if scores[d] < minimum:
            blocking.append(f"{SPECS[d].label} score {scores[d]} is below the critical minimum {minimum}")
    for d in DIMENSIONS:
        a = getattr(out, d)
        if a.critical_flaw and a.confidence >= 0.5:
            blocking.append(f"{SPECS[d].label}: {a.critical_flaw_reason or 'critical flaw identified'}")

    reasons = [f"Weighted score {weighted} vs threshold {CONFIG['publishable_threshold']}"]
    threshold = CONFIG["publishable_threshold"]
    if blocking:
        decision = NON_PUBLISHABLE
        reasons.append("Critical-failure rule triggered")
    elif insufficient_input:
        decision = HUMAN_REVIEW
        reasons.append("Extracted paper content is insufficient for an automated decision")
    elif confidence < CONFIG["min_confidence"]:
        decision = HUMAN_REVIEW
        reasons.append(f"Mean confidence {confidence} below {CONFIG['min_confidence']}")
    elif abs(weighted - threshold) < CONFIG["human_review_band"]:
        decision = HUMAN_REVIEW
        reasons.append(f"Borderline: within ±{CONFIG['human_review_band']} of the threshold")
    else:
        decision = PUBLISHABLE if weighted >= threshold else NON_PUBLISHABLE

    if decision == HUMAN_REVIEW:
        policy = CONFIG["human_review_csv_policy"]
        csv_label = int(weighted >= threshold) if policy == "score_threshold" else 0
        reasons.append(f"CSV label from submission policy '{policy}': {csv_label}")
    else:
        csv_label = int(decision == PUBLISHABLE)
    return Decision(decision=decision, weighted_score=weighted, confidence=confidence, blocking_issues=blocking,
                    reasons=reasons, csv_publishable=csv_label, assessment_source=source)
