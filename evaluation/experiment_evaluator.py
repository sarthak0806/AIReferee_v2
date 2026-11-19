"""Experimental validation, results/claims and reproducibility dimensions."""
from __future__ import annotations

import re

from evaluation.methodology_evaluator import doc_text
from evaluation.rubric import DimensionSpec, clamp

# Claims that are rarely realistic and deserve scrutiny (signals, not verdicts).
_EXTREME = re.compile(r"(100(\.0+)?\s?%\s*(accuracy|precision|recall|f1)|perfect (accuracy|score)|"
                      r"outperforms? all (existing|previous|state)|(?<!\d)(\d{3,})\s?% (improvement|better|faster))", re.I)
_STATS = re.compile(r"(±|\\pm|standard deviation|std\.?|confidence interval|p\s?[<=]\s?0\.\d|"
                    r"significan(t|ce) test|random seeds?|averaged over \d+ runs|\d+ runs)", re.I)


# ---------------------------------------------------------------- experimental
def exp_signals(doc, analyzed, evidence, external) -> dict:
    exp = doc.section_text("Experiments") + doc.section_text("Results")
    return {
        "experiment_section_words": len(exp.split()),
        "datasets": analyzed.profile.datasets,
        "baselines": analyzed.profile.baselines,
        "metrics": analyzed.profile.metrics,
        "mentions_ablation": bool(re.search(r"ablation", exp, re.I)),
        "reports_variance_or_significance": bool(_STATS.search(exp)),
        "table_mentions": len(re.findall(r"\bTable\s?\d", exp)),
    }


def exp_fallback(s: dict) -> float:
    score = 3.0 + min(s["experiment_section_words"] / 700, 2.5)
    score += 1.0 if s["datasets"] else 0
    score += 1.0 if s["baselines"] else 0
    score += 1.0 if s["mentions_ablation"] else 0
    score += 1.0 if s["reports_variance_or_significance"] else 0
    return clamp(score)


EXPERIMENTAL = DimensionSpec(
    name="experimental", label="Experimental Validation",
    questions=["Are the experiments sufficient to support the claims?", "Are datasets and metrics appropriate?",
               "Are comparisons with baselines fair?", "Are ablations and variance/significance reported?"],
    signals=exp_signals, fallback_score=exp_fallback,
)


# --------------------------------------------------------------------- results
def results_signals(doc, analyzed, evidence, external) -> dict:
    text = doc_text(doc)
    return {
        "extreme_claims": [m.group(0) for m in _EXTREME.finditer(text)][:5],
        "acknowledges_limitations": bool(doc.section_text("Limitations")) or bool(
            re.search(r"\blimitation", text, re.I)),
        "has_conclusion": bool(doc.section_text("Conclusion")),
        "paper_limitations": analyzed.profile.limitations,
    }


def results_fallback(s: dict) -> float:
    score = 6.0 - 1.5 * len(s["extreme_claims"])
    score += 1.0 if s["acknowledges_limitations"] else -1.0
    score += 0.5 if s["has_conclusion"] else -0.5
    return clamp(score)


RESULTS = DimensionSpec(
    name="results", label="Results and Claims",
    questions=["Do the reported results support the conclusions?", "Are the improvements realistic?",
               "Are limitations acknowledged?", "Are there unsubstantiated claims?"],
    signals=results_signals, fallback_score=results_fallback,
)


# -------------------------------------------------------------- reproducibility
def repro_signals(doc, analyzed, evidence, external) -> dict:
    text = doc_text(doc)
    return {
        "code_link": bool(re.search(r"(github\.com|gitlab\.com|code (is|will be) (publicly )?available|"
                                    r"huggingface\.co)", text, re.I)),
        "hyperparameters": bool(re.search(r"(learning rate|batch size|hyper-?parameter|epochs?|optimizer)", text, re.I)),
        "implementation_details": bool(re.search(r"implementation details|experimental setup|training details",
                                                 text, re.I)),
        "seeds_or_hardware": bool(re.search(r"(random seed|GPU|TPU|A100|V100|hardware)", text, re.I)),
    }


def repro_fallback(s: dict) -> float:
    return clamp(3.0 + 2.0 * s["code_link"] + 2.0 * s["hyperparameters"] + 1.5 * s["implementation_details"]
                 + 1.0 * s["seeds_or_hardware"])


REPRODUCIBILITY = DimensionSpec(
    name="reproducibility", label="Reproducibility",
    questions=["Are implementation details available?", "Are datasets and parameters described?",
               "Could the experiments be reproduced from the paper?"],
    signals=repro_signals, fallback_score=repro_fallback,
)
