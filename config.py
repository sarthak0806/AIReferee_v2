"""Central configuration for AIReferee.

Every pipeline module reads its parameters from here. To scale the reference
corpus, change ``papers_per_conference`` (e.g. 20 -> 334) and re-run the
ingestion pipeline; existing papers are skipped automatically.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Quiet third-party console noise. Must be set before transformers/huggingface_hub
# are imported, and every module imports config first.
for _k, _v in {
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",   # Windows without Developer Mode: cache copies instead of symlinks
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_VERBOSITY": "error",
    "TRANSFORMERS_VERBOSITY": "error",       # e.g. "sequence longer than max length" while counting tokens
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(_k, _v)

CONFIG = {
    # Conference configuration
    "conferences": ["cvpr", "neurips", "daa", "emnlp", "tmlr", "kdd"],

    # Conferences kept in the output schema but excluded from collection and
    # matching (DAA: acronym unverified, skipped by user decision).
    "skip_conferences": ["daa"],

    # Main scalability variable
    "papers_per_conference": 20,

    # Optional maximum corpus size (None = papers_per_conference * len(conferences))
    "max_total_papers": None,

    # Ingestion
    "download_pdfs": True,
    "skip_existing_papers": True,
    "incremental_ingestion": True,
    "min_pdf_pages": 4,            # shorter PDFs are rejected as non-papers
    "min_extractable_chars": 2000,  # text needed on first pages to count as valid
    "max_pdf_mb": 60,
    "http_timeout_s": 60,
    "http_retries": 3,
    "request_delay_s": 1.0,         # polite delay between requests to one host
    "arxiv_request_delay_s": 3.0,   # arXiv API terms ask for >= 3 s between calls

    # Chunking (tokens are counted with the embedding model's own tokenizer)
    "chunk_size": 500,
    "chunk_overlap": 100,
    "min_chunk_tokens": 40,                       # drop tiny fragments
    "chunk_exclude_sections": ["References", "Acknowledgements", "Front Matter"],

    # Embeddings
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_batch_size": 32,
    "embedding_max_seq_length": 512,              # chunks of 500 tokens must fit
    "benchmark_embedding_models": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "nomic-ai/nomic-embed-text-v1",
    ],

    # FAISS
    "faiss_index_type": "flat_ip",                # flat_ip | flat_l2 | ivf_flat
    "faiss_ivf_nlist": 100,

    # Retrieval
    "top_k_similarity": 30,
    "top_k_reranking": 10,
    "final_evidence_count": 5,
    "max_evidence_per_paper": 2,
    "relevance_thresholds": {"direct": 2.0, "indirect": -2.0},  # cross-encoder logits

    # Cross-Encoder
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "benchmark_reranker_models": ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"],

    # External search (metadata-only results; never treated as full-text evidence)
    "enable_arxiv": True,
    "enable_google_scholar": True,                # requires SERPAPI_API_KEY
    "max_scholar_queries_per_paper": 2,           # SerpAPI free plan = 250 searches/month
    "external_search_cache": True,                # reuse identical query results (saves quota)
    "external_results_per_query": 5,
    "fuzzy_title_threshold": 92,

    # LLM (Groq). Disabled until GROQ_API_KEY is set; local fallbacks are used
    # and every output records which path produced it.
    "enable_llm": True,                           # still requires GROQ_API_KEY
    "llm_provider": "groq",
    "llm_model": "openai/gpt-oss-120b",           # llama-3.3-70b no longer served by Groq
    # Each Groq model has its own free-tier quota; when the primary's daily quota
    # is exhausted the next model is used (the model is recorded in every output).
    "llm_fallback_models": ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    "llm_max_wait_s": 90,                         # longer rate-limit waits switch model instead of blocking
    "llm_max_retries": 3,
    "llm_rate_limit_waits": 8,                    # Groq free tier: 8k tokens/minute
    # Groq counts input + max_tokens per request against tokens-per-minute. On the
    # free tier (8k TPM) one request must stay below this; raise it on paid tiers.
    "llm_request_token_budget": 7600,
    "llm_chars_per_token": 3.2,                   # conservative estimate for budgeting
    "llm_reasoning_effort": "low",                # gpt-oss reasoning budget (low|medium|high)
    "llm_temperature": 0.0,
    "llm_max_input_chars": 16000,                 # keeps one call under the free-tier TPM
    "num_queries": 6,

    # Publishability scoring (starting assumptions, NOT validated standards)
    "dimension_weights": {
        "novelty": 0.25, "methodology": 0.25, "experimental": 0.20,
        "literature": 0.10, "results": 0.10, "reproducibility": 0.10,
    },
    "publishable_threshold": 6.0,
    "human_review_band": 0.5,                     # |score - threshold| < band -> review
    "critical_min_scores": {"methodology": 3.0, "experimental": 3.0, "results": 3.0},
    "min_confidence": 0.4,
    # Submission policy for HUMAN_REVIEW in the hackathon CSV (binary labels only):
    #   "score_threshold": use the side of the threshold the weighted score falls on
    #   "non_publishable": conservative, never recommend an unresolved paper
    # The internal report always keeps the HUMAN_REVIEW status.
    "human_review_csv_policy": "score_threshold",

    # Conference matching signal weights
    "conference_signal_weights": {"centroid": 0.25, "neighbours": 0.30, "keywords": 0.15, "llm": 0.30},
    "conference_neighbour_k": 10,                 # FAISS neighbours per uploaded-paper chunk
    "conference_max_probe_chunks": 40,
    "conference_profile_keywords": 25,

    # Output
    "max_rationale_words": 100,

    # Storage (relative to PROJECT_ROOT)
    "database_path": "data/aireferee.db",
    "faiss_index_path": "data/faiss/reference.index",
    "reference_papers_dir": "data/reference_papers",
    "evaluation_dataset_dir": "data/evaluation_dataset",
    "extracted_dir": "data/extracted",
    "reports_dir": "data/reports",
    "log_dir": "logs",
}

# ---------------------------------------------------------------------------
# Conference sources.
#
# Sources are consumed in order until the per-conference target is met, so
# adding more years/volumes here is how the corpus scales past one
# proceedings volume. Every source type verifies venue membership:
#   cvf / neurips / acl  -> official proceedings (membership is by construction)
#   arxiv                -> only papers whose author-supplied comment or
#                           journal_ref states acceptance/publication at the venue
# ---------------------------------------------------------------------------
CONFERENCE_SOURCES = {
    "cvpr": [
        {"type": "cvf", "event": "CVPR2024", "year": 2024},
        {"type": "cvf", "event": "CVPR2023", "year": 2023},
    ],
    "neurips": [
        {"type": "neurips", "year": 2023},
        {"type": "neurips", "year": 2022},
    ],
    "emnlp": [
        {"type": "acl", "volume": "2023.emnlp-main", "year": 2023},
        {"type": "acl", "volume": "2022.emnlp-main", "year": 2022},
    ],
    "tmlr": [
        {
            "type": "arxiv",
            "query": 'co:"Transactions on Machine Learning Research" OR jr:"Transactions on Machine Learning Research"',
            "accept_patterns": [
                r"published\s+in\s+(the\s+)?transactions\s+on\s+machine\s+learning\s+research",
                r"(accepted|appear\w*|published)\b.{0,40}\b(tmlr|transactions\s+on\s+machine\s+learning\s+research)",
                r"(tmlr|transactions\s+on\s+machine\s+learning\s+research)\b.{0,15}\b(20\d\d|\(\d\d/20\d\d\))",
            ],
        },
    ],
    "kdd": [
        {
            "type": "arxiv",
            "query": 'co:"KDD 2024"', "year": 2024,
            "accept_patterns": [
                r"(accepted|appear\w*|published|camera[- ]ready|proceedings)\b.{0,60}\b(sig)?kdd\s*'?(20)?24",
                r"\b(sig)?kdd\s*'?(20)?24\b.{0,30}\b(accepted|camera[- ]ready)",
            ],
        },
        {
            "type": "arxiv",
            "query": 'co:"KDD 2023"', "year": 2023,
            "accept_patterns": [
                r"(accepted|appear\w*|published|camera[- ]ready|proceedings)\b.{0,60}\b(sig)?kdd\s*'?(20)?23",
                r"\b(sig)?kdd\s*'?(20)?23\b.{0,30}\b(accepted|camera[- ]ready)",
            ],
        },
    ],
    # The acronym "DAA" is kept exactly as given in the hackathon problem
    # statement. Its intended full name is not yet verified, so no source is
    # configured and the collector reports a shortfall instead of guessing.
    "daa": [],
}

# Phrases that disqualify an arXiv comment as venue evidence.
ARXIV_REJECT_PATTERNS = [
    r"\bsubmitted\b", r"under\s+review", r"\breject", r"\bworkshop\b",
    r"in\s+submission", r"\bpreprint\b.{0,20}\bsubmitted",
]

CONFERENCE_INFO = {
    "cvpr": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    "neurips": "Conference on Neural Information Processing Systems",
    "daa": "DAA (full name pending verification)",
    "emnlp": "Conference on Empirical Methods in Natural Language Processing",
    "tmlr": "Transactions on Machine Learning Research",
    "kdd": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
}

# Secrets come from the environment only.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")


def path(key: str) -> Path:
    """Resolve a configured storage path relative to the project root."""
    return PROJECT_ROOT / CONFIG[key]


def active_conferences() -> list[str]:
    """Conferences that are collected, indexed and eligible for recommendation."""
    return [c for c in CONFIG["conferences"] if c not in CONFIG["skip_conferences"]]


def conference_dir(conference: str) -> Path:
    return path("reference_papers_dir") / conference


def target_for(conference: str) -> int:
    """Per-conference target, respecting the optional global cap."""
    per = CONFIG["papers_per_conference"]
    cap = CONFIG["max_total_papers"]
    if cap is None:
        return per
    return min(per, cap // len(CONFIG["conferences"]))


def ensure_directories() -> None:
    for conf in CONFIG["conferences"]:
        conference_dir(conf).mkdir(parents=True, exist_ok=True)
    (path("evaluation_dataset_dir") / "papers").mkdir(parents=True, exist_ok=True)
    (path("extracted_dir") / "papers").mkdir(parents=True, exist_ok=True)
    (path("extracted_dir") / "chunks").mkdir(parents=True, exist_ok=True)
    path("faiss_index_path").parent.mkdir(parents=True, exist_ok=True)
    path("reports_dir").mkdir(parents=True, exist_ok=True)
    path("log_dir").mkdir(parents=True, exist_ok=True)
