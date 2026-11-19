"""AIReferee Streamlit application.

Run from the AIReferee/ directory:  streamlit run app.py
"""
from __future__ import annotations

import os

# Progress is shown in the UI; keep only warnings and errors in the console (full log: logs/aireferee.log).
os.environ.setdefault("AIREFEREE_CONSOLE_LOG", "WARNING")

import json
import sqlite3
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from config import CONFERENCE_INFO, CONFIG, active_conferences, path
from database.db_manager import DatabaseManager
from evaluation.publishability_classifier import SPECS
from evaluation.rubric import DIMENSIONS
from retrieval.scholar_search import scholar_available
from utils.llm import llm_available

st.set_page_config(page_title="AIReferee", page_icon="📄", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1280px;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .ar-banner {border-radius: 10px; padding: 14px 18px; margin: 4px 0 12px 0; border: 1px solid;}
  .ar-banner h3 {margin: 0 0 4px 0; font-size: 1.25rem;}
  .ar-banner p {margin: 0; opacity: .85;}
  .ar-pub {background: rgba(46,160,67,.10); border-color: rgba(46,160,67,.55);}
  .ar-non {background: rgba(218,54,51,.10); border-color: rgba(218,54,51,.55);}
  .ar-rev {background: rgba(210,153,34,.12); border-color: rgba(210,153,34,.6);}
  .ar-step {font-size: .85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

DECISION_STYLE = {
    "PUBLISHABLE": ("ar-pub", "Publishable"),
    "NON-PUBLISHABLE": ("ar-non", "Non-publishable"),
    "HUMAN_REVIEW": ("ar-rev", "Human review"),
}
RELEVANCE_COLOUR = {"directly relevant": "green", "indirectly relevant": "orange", "weakly relevant": "gray"}


# ------------------------------------------------------------------ data
@st.cache_resource
def get_db() -> DatabaseManager:
    db = DatabaseManager()
    # Streamlit reruns scripts on worker threads; the connection must allow that.
    db.conn.close()
    db.conn = sqlite3.connect(db.db_path, check_same_thread=False)
    db.conn.row_factory = sqlite3.Row
    db.conn.execute("PRAGMA foreign_keys = ON")
    return db


def corpus_stats(db: DatabaseManager) -> pd.DataFrame:
    rows = []
    for conf in CONFIG["conferences"]:
        q = lambda sql: db.conn.execute(sql, (conf,)).fetchone()[0]
        rows.append({
            "Conference": conf.upper(),
            "Status": "skipped" if conf in CONFIG["skip_conferences"] else "active",
            "Target": CONFIG["papers_per_conference"] if conf in active_conferences() else 0,
            "Valid PDFs": q("SELECT COUNT(*) FROM papers WHERE conference=? AND download_status='valid'"),
            "Failed": q("SELECT COUNT(*) FROM papers WHERE conference=? AND download_status='failed'"),
            "Invalid / OCR": q("SELECT COUNT(*) FROM papers WHERE conference=? "
                               "AND download_status IN ('invalid','needs_ocr')"),
            "Metadata only": q("SELECT COUNT(*) FROM papers WHERE conference=? AND download_status='metadata_only'"),
            "Extracted": q("SELECT COUNT(*) FROM papers WHERE conference=? AND extraction_status='extracted'"),
            "Chunks": q("SELECT COUNT(*) FROM chunks WHERE conference=?"),
            "Embedded": q("SELECT COUNT(*) FROM chunks WHERE conference=? AND embedding_status='embedded'"),
        })
    return pd.DataFrame(rows)


def faiss_status(db: DatabaseManager) -> dict:
    f = path("faiss_index_path")
    n_map = db.conn.execute("SELECT COUNT(*) FROM vector_map WHERE embedding_model=?",
                            (CONFIG["embedding_model"],)).fetchone()[0]
    return {"exists": f.exists(), "size_mb": round(f.stat().st_size / 2**20, 2) if f.exists() else 0,
            "vectors": n_map}


def load_results() -> list[dict]:
    reports = sorted(path("reports_dir").glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in reports:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


# ------------------------------------------------------------- components
def decision_banner(res: dict) -> None:
    d = res["decision"]
    cls, label = DECISION_STYLE[d["decision"]]
    conf = res.get("conference")
    venue = f" · recommended venue <b>{conf['recommended'].upper()}</b>" if conf else ""
    reason = d["reasons"][-1] if d["reasons"] else ""
    st.markdown(f"<div class='ar-banner {cls}'><h3>{label}{venue}</h3>"
                f"<p>Weighted score {d['weighted_score']} / 10 (threshold {CONFIG['publishable_threshold']}) · "
                f"confidence {d['confidence']} · CSV label {d['csv_publishable']} · {reason}</p></div>",
                unsafe_allow_html=True)


def dimension_chart(res: dict) -> alt.Chart:
    a = res["assessment"]
    df = pd.DataFrame([{"Dimension": SPECS[x].label, "Score": a[x]["score"],
                        "Weight": f"{int(CONFIG['dimension_weights'][x] * 100)}%",
                        "Confidence": a[x]["confidence"], "Critical": a[x]["critical_flaw"]} for x in DIMENSIONS])
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 10]), title="Score (0–10)"),
        y=alt.Y("Dimension:N", sort=None, title=None),
        color=alt.condition(alt.datum.Score >= CONFIG["publishable_threshold"], alt.value("#4c78a8"),
                            alt.value("#e4a33a")),
        tooltip=["Dimension", "Score", "Weight", "Confidence", "Critical"])
    rule = alt.Chart(pd.DataFrame({"t": [CONFIG["publishable_threshold"]]})).mark_rule(
        strokeDash=[4, 4], color="#888").encode(x="t:Q")
    return (bars + rule).properties(height=220)


def conference_chart(conf: dict) -> alt.Chart:
    df = pd.DataFrame([{"Venue": r["conference"].upper(), "Fit score": r["score"],
                        "Chosen": r["conference"] == conf["recommended"]} for r in conf["ranking"]])
    return alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Fit score:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Venue:N", sort="-x", title=None),
        color=alt.condition(alt.datum.Chosen, alt.value("#4c78a8"), alt.value("#bbbbbb")),
        tooltip=["Venue", "Fit score"]).properties(height=180)


def bullet_list(items: list[str], empty: str = "None recorded.") -> None:
    st.markdown("\n".join(f"- {x}" for x in items) if items else f"_{empty}_")


def show_result(res: dict, key: str) -> None:
    """Render one evaluation. ``key`` makes every widget id unique on the page."""
    prof = res["analysis"]["profile"]
    st.subheader(prof["title"] or res["paper_id"])
    st.caption(f"Paper ID {res['paper_id']} · {res['pdf_name']} · evaluated {res['evaluated_at'][:19]} UTC · "
               f"{res['analysis']['page_count']} pages")
    decision_banner(res)

    t_sum, t_prof, t_ret, t_eval, t_conf, t_audit = st.tabs(
        ["Summary", "Paper profile", "Retrieval", "Evaluation", "Conference", "Audit trail"])

    with t_sum:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Dimension scores**")
            st.altair_chart(dimension_chart(res), width="stretch")
        with right:
            st.markdown("**Overall assessment**")
            st.write(res["assessment"]["overall_summary"] or "_No summary._")
            if res["decision"]["blocking_issues"]:
                st.error("Blocking issues:\n\n" + "\n".join(f"- {b}" for b in res["decision"]["blocking_issues"]))
        if res.get("conference"):
            st.info(f"**{res['conference']['recommended'].upper()}** — {res['conference']['rationale']}")

    with t_prof:
        st.caption(f"Profile source: {res['analysis']['profile_source']} · sections detected: "
                   f"{', '.join(res['analysis']['sections_found'])}")
        st.markdown(f"**Abstract.** {prof['abstract']}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Research problem.** {prof['research_problem'] or '—'}")
            st.markdown("**Claimed contributions**")
            bullet_list(prof["claimed_contributions"])
            st.markdown(f"**Methodology.** {prof['methodology'] or '—'}")
        with c2:
            st.markdown(f"**Datasets:** {', '.join(prof['datasets']) or '—'}")
            st.markdown(f"**Baselines:** {', '.join(prof['baselines']) or '—'}")
            st.markdown(f"**Metrics:** {', '.join(prof['metrics']) or '—'}")
            st.markdown(f"**Results.** {prof['results'] or '—'}")
            st.markdown("**Stated limitations**")
            bullet_list(prof["limitations"])

    with t_ret:
        st.markdown("**Generated queries** "
                    f"<span class='ar-step'>({res['query_source']})</span>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(res["queries"])[["category", "query", "reason"]], hide_index=True,
                     width="stretch", key=f"{key}-queries")
        st.markdown("**Selected evidence** — full-text passages after Cross-Encoder reranking")
        for e in res["evidence"]:
            with st.container(border=True):
                colour = RELEVANCE_COLOUR.get(e["relevance"], "gray")
                st.markdown(f"**{e['evidence_id']} · {e['title']}**  \n"
                            f":{colour}-badge[{e['relevance']}] "
                            f":blue-badge[{(e.get('conference') or '').upper()} {e.get('year') or ''}] "
                            f"section {e['section']}, p.{e['page']} · bi-encoder {e['similarity_score']} · "
                            f"reranker {e['reranker_score']}")
                st.caption(e["passage"][:700] + ("…" if len(e["passage"]) > 700 else ""))
                if e.get("source_url"):
                    st.caption(e["source_url"])
        with st.expander("Internal FAISS results per query (stage 1, bi-encoder)"):
            for q, hits in zip(res["queries"], res["internal_results"]):
                st.markdown(f"_{q['category']}_: {q['query']}")
                st.dataframe(pd.DataFrame(hits), hide_index=True, width="stretch",
                             key=f"{key}-faiss-{q['query_id']}")
        with st.expander(f"External literature — arXiv and Google Scholar ({len(res['external_results'])}, "
                         "metadata only)"):
            if res["external_results"]:
                ext = pd.DataFrame(res["external_results"])
                cols = [c for c in ["evidence_id", "title", "year", "sources", "reranker_score", "relevance",
                                    "cited_by", "in_internal_corpus", "url"] if c in ext]
                st.dataframe(ext[cols], hide_index=True, width="stretch", key=f"{key}-external")
            else:
                st.write("No external results (disabled or no matches).")

    with t_eval:
        a = res["assessment"]
        d = res["decision"]
        st.caption(f"Assessment source: {d['assessment_source']}")
        for r in d["reasons"]:
            st.markdown(f"- {r}")
        for x in DIMENSIONS:
            dim = a[x]
            flag = " · critical flaw" if dim["critical_flaw"] else ""
            with st.expander(f"{SPECS[x].label} — {dim['score']}/10 · confidence {dim['confidence']}{flag}"):
                st.write(dim["finding"])
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Strengths**")
                    bullet_list(dim["strengths"])
                with c2:
                    st.markdown("**Weaknesses**")
                    bullet_list(dim["weaknesses"])
                st.markdown(f"**Supporting evidence:** {', '.join(dim['supporting_evidence']) or '—'} · "
                            f"**Counter-evidence:** {', '.join(dim['counter_evidence']) or '—'}")
                st.markdown("**Missing evidence**")
                bullet_list(dim["missing_evidence"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Unsubstantiated claims**")
            bullet_list(a["unsubstantiated_claims"])
            st.markdown("**Coherence issues**")
            bullet_list(a["coherence_issues"])
        with c2:
            st.markdown("**Revision roadmap**")
            bullet_list(a["revision_roadmap"])

    with t_conf:
        conf = res.get("conference")
        if not conf:
            st.info("No conference recommendation: the CSV label is 0 (not publishable), so Conference and "
                    "Rationale are `na`.")
        else:
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown(f"#### {conf['recommended'].upper()}")
                st.caption(CONFERENCE_INFO[conf["recommended"]])
                st.altair_chart(conference_chart(conf), width="stretch")
            with c2:
                st.markdown("**Rationale**")
                st.write(conf["rationale"])
                st.caption(f"{len(conf['rationale'].split())} words (limit {CONFIG['max_rationale_words']}) · "
                           f"source: {conf['rationale_source']}")
                st.markdown("**Closest reference papers at this venue**")
                for p in conf["ranking"][0]["top_reference_papers"][:4]:
                    st.markdown(f"- {p['title']} — similarity {p['similarity_score']} ({p['section']})")
            st.markdown("**Signals per venue** (weights: " +
                        ", ".join(f"{k} {v:.0%}" for k, v in conf["weights_used"].items()) + ")")
            st.dataframe(pd.DataFrame([{"Venue": r["conference"].upper(), "Fit score": r["score"], **r["signals"],
                                        "LLM reason": r.get("llm_reason")} for r in conf["ranking"]]),
                         hide_index=True, width="stretch", key=f"{key}-signals")

    with t_audit:
        st.markdown("**Configuration used**")
        st.json(res["config"], expanded=False)
        st.markdown("**Timings (seconds)**")
        st.bar_chart(pd.Series(res["timings_s"]), horizontal=True, height=200)
        if res.get("excluded_self_matches"):
            st.caption(f"Excluded from retrieval (same file in corpus): {res['excluded_self_matches']}")

    from evaluation.final_evaluator import csv_row
    from reporting.csv_generator import to_csv_string
    from reporting.pdf_report import build_pdf
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download CSV row", to_csv_string([csv_row(res)]), f"{res['paper_id']}_result.csv",
                       "text/csv", key=f"{key}-dl-csv", width="stretch")
    c2.download_button("Download PDF report", build_pdf(res), f"{res['paper_id']}_report.pdf",
                       "application/pdf", key=f"{key}-dl-pdf", width="stretch")
    c3.download_button("Download full JSON", json.dumps(res, indent=2, default=str), f"{res['paper_id']}.json",
                       "application/json", key=f"{key}-dl-json", width="stretch")


def run_pipeline(files, use_ext: bool, ids: list[str | None]) -> list[dict]:
    from evaluation.final_evaluator import evaluate_paper
    from reporting.pdf_report import build_pdf
    results = []
    db = get_db()
    for i, (up, pid) in enumerate(zip(files, ids), 1):
        status = st.status(f"[{i}/{len(files)}] {up.name}", expanded=len(files) == 1)
        with tempfile.TemporaryDirectory(dir=path("reports_dir")) as tmp:
            f = Path(tmp) / up.name
            f.write_bytes(up.getvalue())
            try:
                res = evaluate_paper(f, paper_id=pid, db=db, use_external=use_ext,
                                     progress=lambda step, msg: status.write(f"**{step}** — {msg}"))
                (path("reports_dir") / res["paper_id"] / "report.pdf").write_bytes(build_pdf(res))
                status.update(label=f"[{i}/{len(files)}] {up.name} — {res['decision']['decision']}",
                              state="complete", expanded=False)
                results.append(res)
            except Exception as exc:
                status.update(label=f"[{i}/{len(files)}] {up.name} — failed", state="error")
                status.exception(exc)
    return results


# ------------------------------------------------------------------ pages
def page_dashboard(db: DatabaseManager) -> None:
    st.title("AIReferee")
    st.caption("Research-paper publishability classification and conference recommendation")
    stats = corpus_stats(db)
    fs = faiss_status(db)
    results = load_results()
    c = st.columns(5)
    c[0].metric("Reference PDFs", int(stats["Valid PDFs"].sum()),
                help=f"Target {int(stats['Target'].sum())} ({CONFIG['papers_per_conference']} per active venue)")
    c[1].metric("Chunks", int(stats["Chunks"].sum()))
    c[2].metric("FAISS vectors", fs["vectors"], help=f"{fs['size_mb']} MB · {CONFIG['faiss_index_type']}")
    c[3].metric("Papers evaluated", len(results))
    c[4].metric("Publishable (CSV)", sum(r["decision"]["csv_publishable"] for r in results))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Reference corpus by venue**")
        chart_df = stats[stats["Status"] == "active"][["Conference", "Valid PDFs", "Chunks"]]
        st.altair_chart(alt.Chart(chart_df).mark_bar(cornerRadiusEnd=3, color="#4c78a8").encode(
            x=alt.X("Valid PDFs:Q", title="Valid PDFs"), y=alt.Y("Conference:N", sort=None, title=None),
            tooltip=["Conference", "Valid PDFs", "Chunks"]).properties(height=200), width="stretch")
    with right:
        st.markdown("**Recent evaluations**")
        if results:
            for r in results[:5]:
                d = r["decision"]
                venue = (r.get("conference") or {}).get("recommended", "na")
                st.markdown(f"- `{r['paper_id']}` · {DECISION_STYLE[d['decision']][1]} · "
                            f"score {d['weighted_score']} · venue {str(venue).upper()}")
        else:
            st.write("_No evaluations yet — go to **Evaluate papers**._")
    st.markdown("**Pipeline status**")
    s = st.columns(4)
    s[0].markdown(f"Embedding model  \n`{CONFIG['embedding_model']}`")
    s[1].markdown(f"Reranker  \n`{CONFIG['reranker_model']}`")
    s[2].markdown(f"LLM  \n`{CONFIG['llm_model']}` " + ("✅" if llm_available() else "⚠️ fallback"))
    s[3].markdown("External search  \n" + f"arXiv {'✅' if CONFIG['enable_arxiv'] else 'off'} · "
                  f"Scholar {'✅' if scholar_available() else 'off'}")


def page_evaluate(db: DatabaseManager) -> None:
    st.title("Evaluate papers")
    if not llm_available():
        st.warning("Groq LLM unavailable: profiles, queries and assessments use labelled heuristic fallbacks "
                   "and are routed to Human Review.")
    files = st.file_uploader("Upload one or more research papers (PDF)", type=["pdf"], accept_multiple_files=True,
                             key="eval-upload")
    c1, c2 = st.columns([2, 1])
    single_id = c1.text_input("Paper ID (single upload only; default = file name)", key="eval-id",
                              disabled=bool(files) and len(files) > 1)
    use_ext = c2.toggle("Search arXiv + Google Scholar", value=True, key="eval-ext",
                        help=f"Uses up to {CONFIG['max_scholar_queries_per_paper']} SerpAPI searches per paper "
                             "(free plan: 250 a month). Repeated queries come from the cache.")
    if files:
        st.caption(f"{len(files)} file(s) · about 2 minutes each on the free Groq tier.")
    if st.button("Run evaluation", type="primary", disabled=not files, key="eval-run"):
        ids = [single_id.strip() or None] if len(files) == 1 else [Path(f.name).stem for f in files]
        if len(files) == 1 and ids[0] is None:
            ids = [Path(files[0].name).stem]
        st.session_state["last_results"] = run_pipeline(files, use_ext, ids)
    last = st.session_state.get("last_results") or []
    if len(last) == 1:
        show_result(last[0], key="eval-last")
    elif len(last) > 1:
        from evaluation.final_evaluator import csv_row
        st.dataframe(pd.DataFrame([csv_row(r) for r in last]), hide_index=True, width="stretch", key="eval-batch")
        st.info("Open any paper in **Results**.")


def page_results(db: DatabaseManager) -> None:
    from evaluation.final_evaluator import csv_row
    from reporting.csv_generator import to_csv_string
    st.title("Results")
    results = load_results()
    if not results:
        st.info("No evaluations yet.")
        return
    table = pd.DataFrame([{**csv_row(r), "Decision": DECISION_STYLE[r["decision"]["decision"]][1],
                           "Score": r["decision"]["weighted_score"], "Confidence": r["decision"]["confidence"],
                           "Evaluated": r["evaluated_at"][:16]} for r in results])
    f1, f2 = st.columns(2)
    dec = f1.multiselect("Decision", sorted(table["Decision"].unique()), key="res-filter-decision")
    ven = f2.multiselect("Conference", sorted(table["Conference"].unique()), key="res-filter-venue")
    mask = pd.Series(True, index=table.index)
    if dec:
        mask &= table["Decision"].isin(dec)
    if ven:
        mask &= table["Conference"].isin(ven)
    view = table[mask]
    st.dataframe(view, hide_index=True, width="stretch", key="res-table",
                 column_config={"Rationale": st.column_config.TextColumn(width="large"),
                                "Score": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.2f")})
    rows = sorted((csv_row(r) for r in results), key=lambda r: r["Paper ID"])
    st.download_button("Download results.csv (hackathon format, all papers)", to_csv_string(rows), "results.csv",
                       "text/csv", key="res-dl-all", type="primary")
    st.divider()
    pick = st.selectbox("Open an evaluation", [r["paper_id"] for r in results], key="res-pick")
    if pick:
        show_result(next(r for r in results if r["paper_id"] == pick), key=f"res-{pick}")


def page_corpus(db: DatabaseManager) -> None:
    from ingestion import ingestion_pipeline as ip
    st.title("Reference corpus")
    st.write(f"Target: **{CONFIG['papers_per_conference']}** papers per active venue "
             f"(`papers_per_conference` in `config.py`). DAA is skipped until its full name is verified.")
    st.dataframe(corpus_stats(db), hide_index=True, width="stretch", key="corpus-stats")
    c1, c2, c3 = st.columns(3)
    if c1.button("Collect missing PDFs", key="corpus-collect", width="stretch"):
        with st.spinner("Collecting (rate-limited)…"):
            st.json(ip.collect()["conferences"])
    if c2.button("Extract and chunk new papers", key="corpus-process", width="stretch"):
        with st.spinner("Extracting…"):
            st.json(ip.process())
    if c3.button("Embed and index new chunks", key="corpus-index", width="stretch"):
        with st.spinner("Embedding…"):
            st.json(ip.index())
    venue = st.selectbox("Venue", [c.upper() for c in active_conferences()], key="corpus-venue")
    papers = db.conn.execute(
        "SELECT paper_id, publication_year AS year, title, source_name AS source, venue_verification, page_count, "
        "extraction_status FROM papers WHERE download_status='valid' AND conference=? ORDER BY paper_id",
        (venue.lower(),)).fetchall()
    st.dataframe(pd.DataFrame([dict(r) for r in papers]), hide_index=True, width="stretch", key="corpus-papers")
    prof = db.conn.execute("SELECT reference_profile FROM conferences WHERE conference_acronym=?",
                           (venue.lower(),)).fetchone()
    if prof and prof[0]:
        p = json.loads(prof[0])
        st.markdown(f"**Venue profile:** {p['llm_profile'].get('summary') or '—'}")
        st.caption("Top keywords: " + ", ".join(list(p["keywords"])[:15]))
    failed = db.conn.execute("SELECT paper_id, conference, download_status, status_detail, source_url FROM papers "
                             "WHERE download_status <> 'valid' ORDER BY conference").fetchall()
    with st.expander(f"Failed, invalid or metadata-only records ({len(failed)})"):
        st.dataframe(pd.DataFrame([dict(r) for r in failed]), width="stretch", key="corpus-failed")


def page_settings(db: DatabaseManager) -> None:
    st.title("Configuration")
    st.caption("Read-only view of `config.py`. Edit the file and restart the app to change settings.")
    w = CONFIG["dimension_weights"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Decision policy**")
        st.markdown("\n".join([
            f"- Threshold: **{CONFIG['publishable_threshold']}** / 10, review band ±{CONFIG['human_review_band']}",
            f"- Critical minimums: {CONFIG['critical_min_scores']}",
            f"- Minimum confidence: {CONFIG['min_confidence']}",
            f"- Human Review in CSV: `{CONFIG['human_review_csv_policy']}`",
            "- Weights: " + ", ".join(f"{SPECS[k].label} {v:.0%}" for k, v in w.items()),
        ]))
    with c2:
        st.markdown("**Retrieval**")
        st.markdown("\n".join([
            f"- Chunks: {CONFIG['chunk_size']} tokens, overlap {CONFIG['chunk_overlap']}",
            f"- FAISS top-k {CONFIG['top_k_similarity']} → rerank → evidence {CONFIG['final_evidence_count']} "
            f"(max {CONFIG['max_evidence_per_paper']} per paper)",
            f"- Conference signal weights: {CONFIG['conference_signal_weights']}",
            f"- LLM request budget: {CONFIG['llm_request_token_budget']} tokens",
        ]))
    st.json({k: v for k, v in CONFIG.items() if not isinstance(v, (dict, list))}, expanded=False)


# ------------------------------------------------------------------- main
PAGES = {"Dashboard": page_dashboard, "Evaluate papers": page_evaluate, "Results": page_results,
         "Reference corpus": page_corpus, "Configuration": page_settings}

with st.sidebar:
    st.markdown("### 📄 AIReferee")
    page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
    st.divider()
    from utils.llm import quota_status
    st.caption(f"LLM: {'available' if llm_available() else 'unavailable (fallbacks)'}  \n"
               f"Groq quota: {quota_status()}  \n"
               f"Scholar: {'on' if scholar_available() else 'off'} · arXiv: "
               f"{'on' if CONFIG['enable_arxiv'] else 'off'}")

PAGES[page](get_db())
