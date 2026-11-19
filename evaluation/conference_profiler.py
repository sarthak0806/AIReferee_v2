"""Build conference profiles from the reference corpus.

Per active conference:
  * centroid  - mean of its chunk embeddings (from FAISS), L2-normalised
  * keywords  - distinctive TF-IDF terms of its papers vs. the other venues
  * profile   - optional LLM summary of dominant topics, problems, methods,
                contribution types and experimental patterns, grounded in the
                reference paper titles/abstracts
Profiles are stored in the ``conferences`` table; centroids in an .npz file.
"""
from __future__ import annotations

import json
import re

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from config import CONFERENCE_INFO, CONFIG, active_conferences, path
from database.db_manager import DatabaseManager
from embeddings.embedding_model import get_embedding_model
from retrieval.faiss_manager import FaissManager
from utils.llm import LLMUnavailable, complete_json, llm_available, model_tag
from utils.logging_config import get_logger

log = get_logger("conference_profiler")


class VenueProfile(BaseModel):
    dominant_topics: list[str] = Field(default_factory=list)
    common_research_problems: list[str] = Field(default_factory=list)
    frequent_methodologies: list[str] = Field(default_factory=list)
    application_areas: list[str] = Field(default_factory=list)
    typical_contributions: list[str] = Field(default_factory=list)
    experimental_patterns: list[str] = Field(default_factory=list)
    summary: str = ""


PROMPT = """Below are titles and abstracts of {n} papers published at {venue}. Summarise what this reference set
shows about the venue: dominant topics, common research problems, frequently used methodologies, application
areas, typical contribution types (theoretical / methodological / empirical / systems / resource) and
experimental patterns. Describe only what these papers show; do not add outside facts about the venue.
Lists: 4-8 short items each. summary: <= 80 words.

<evidence>
{papers}
</evidence>"""


_LATEX = re.compile(r"\\?(mathbf|mathrm|mathcal|mathbb|boldsymbol|textbf|cdot|ldots|frac|left|right)\b", re.I)
_EXTRA_STOP = {"et", "al", "fig", "figure", "table", "eq", "sec", "section", "appendix", "paper", "proposed",
               "method", "methods", "results", "approach", "use", "using", "used", "show", "shown", "propose",
               "work", "based", "different", "respectively", "wehave", "ofthe", "inthe", "arxiv", "preprint",
               "proceedings", "conference", "http", "https", "github", "com", "org",
               # venue names must not become venue keywords (label leakage)
               "cvpr", "neurips", "nips", "emnlp", "tmlr", "kdd", "sigkdd", "acm", "ieee", "acl", "openreview",
               "information processing", "neural information", "processing systems",
               # glued tokens from PDFs that drop inter-word spaces
               "etal", "thatis", "forexample", "inthe", "ofthe", "inparticular", "inotherwords", "inthiscase",
               "endfor", "endwhile"}


def _sk_stop():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return ENGLISH_STOP_WORDS


def centroids_path():
    return path("faiss_index_path").with_name("conference_centroids.npz")


def paper_vectors(db: DatabaseManager, fm: FaissManager, paper_ids: list[str]) -> dict[str, np.ndarray]:
    """Mean normalised chunk vector per paper, reconstructed from FAISS."""
    out = {}
    for pid in paper_ids:
        vids = [r[0] for r in db.conn.execute(
            "SELECT v.vector_id FROM vector_map v JOIN chunks c ON c.chunk_id=v.chunk_id "
            "WHERE c.paper_id=? AND v.embedding_model=?", (pid, fm.model_name))]
        if vids:
            v = fm.reconstruct(vids).mean(axis=0)
            out[pid] = v / (np.linalg.norm(v) + 1e-12)
    return out


def paper_texts(db: DatabaseManager, conference: str) -> dict[str, str]:
    rows = db.conn.execute(
        "SELECT p.paper_id, p.title, p.abstract, group_concat(c.chunk_text, ' ') FROM papers p "
        "JOIN chunks c ON c.paper_id=p.paper_id WHERE p.conference=? AND p.download_status='valid' "
        "GROUP BY p.paper_id", (conference,)).fetchall()
    return {r[0]: f"{r[1]} {r[2] or ''} {r[3]}" for r in rows}


def venue_keywords(docs: list[str], labels: list[str], top_n: int | None = None) -> dict[str, dict[str, float]]:
    """TF-IDF over individual papers. A term must occur in >= 3 papers (topic-level,
    not one paper's acronym); a venue's keywords are the terms whose mean weight in
    its papers most exceeds their mean weight in the other venues' papers."""
    top_n = top_n or CONFIG["conference_profile_keywords"]
    labels_arr = np.array(labels)
    vec = TfidfVectorizer(stop_words=list(_EXTRA_STOP | set(_sk_stop())), ngram_range=(1, 2), min_df=3,
                          max_df=0.5, max_features=50000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
                          sublinear_tf=True)
    tf = vec.fit_transform([_LATEX.sub(" ", d) for d in docs])
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for c in sorted(set(labels)):
        inside = np.asarray(tf[labels_arr == c].mean(axis=0)).ravel()
        outside = np.asarray(tf[labels_arr != c].mean(axis=0)).ravel()
        row = np.clip(inside - outside, 0, None)
        top = row.argsort()[::-1][:top_n]
        out[c] = {str(vocab[j]): round(float(row[j]), 4) for j in top if row[j] > 0}
    return out


def build_profiles(db: DatabaseManager, use_llm: bool = True) -> dict:
    model = get_embedding_model()
    fm = FaissManager(model.dim, model.name, db).load_or_create()
    confs = [c for c in active_conferences() if db.count_valid(c)]
    docs, labels = [], []
    for c in confs:
        for txt in paper_texts(db, c).values():
            docs.append(txt)
            labels.append(c)
    kw = venue_keywords(docs, labels)

    centroids, profiles = {}, {}
    for conf in confs:
        pids = [p["paper_id"] for p in db.list_papers(conf, "valid")]
        pv = paper_vectors(db, fm, pids)
        c = np.mean(list(pv.values()), axis=0)
        centroids[conf] = c / (np.linalg.norm(c) + 1e-12)
        keywords = kw[conf]

        venue_profile, source = VenueProfile(), "none"
        if use_llm and llm_available():
            papers = "\n".join(f"- {p['title']}: {(p.get('abstract') or '')[:600]}" for p in db.list_papers(conf, "valid"))
            try:
                venue_profile = complete_json(PROMPT.format(n=len(pids), venue=conf.upper(), papers=papers[:20000]),
                                              VenueProfile, max_tokens=2000)
                source = model_tag()
            except LLMUnavailable as exc:
                log.warning("[%s] LLM profile unavailable: %s", conf, exc)
        profile = {"keywords": keywords, "paper_count": len(pids), "embedding_model": model.name,
                   "llm_profile": venue_profile.model_dump(), "llm_profile_source": source}
        profiles[conf] = profile
        db.upsert_conference(conf, CONFERENCE_INFO[conf], description=venue_profile.summary or None,
                             topics=json.dumps(venue_profile.dominant_topics or list(keywords)[:10]),
                             profile=json.dumps(profile))
        log.info("[%s] profile built from %d papers; top terms: %s", conf, len(pids), ", ".join(list(keywords)[:8]))
    np.savez(centroids_path(), **centroids)
    return profiles


def load_profiles(db: DatabaseManager) -> tuple[dict, dict]:
    profiles = {}
    for r in db.conn.execute("SELECT conference_acronym, reference_profile FROM conferences"):
        if r[1] and r[0] in active_conferences():
            profiles[r[0]] = json.loads(r[1])
    cp = centroids_path()
    if not cp.exists() or not profiles:
        raise FileNotFoundError("Conference profiles not built. Run: python -m evaluation.conference_profiler")
    data = np.load(cp)
    return profiles, {k: data[k] for k in data.files}


if __name__ == "__main__":
    d = DatabaseManager()
    res = build_profiles(d)
    print(json.dumps({c: {"papers": p["paper_count"], "top_terms": list(p["keywords"])[:12],
                          "summary": p["llm_profile"]["summary"]} for c, p in res.items()}, indent=2))
