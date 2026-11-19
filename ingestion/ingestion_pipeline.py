"""Incremental reference-corpus pipeline: collect -> process (extract+chunk) -> index (embed+FAISS).

Usage (from the AIReferee/ directory):
    python -m ingestion.ingestion_pipeline build                   # collect + process + index
    python -m ingestion.ingestion_pipeline collect [--conference kdd]
    python -m ingestion.ingestion_pipeline process [--force]
    python -m ingestion.ingestion_pipeline index [--rebuild] [--model NAME]
    python -m ingestion.ingestion_pipeline status
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from config import (CONFIG, CONFERENCE_INFO, PROJECT_ROOT, active_conferences, conference_dir,
                    ensure_directories, path, target_for)
from database.db_manager import DatabaseManager
from ingestion.conference_collector import PaperCandidate, adapters_for
from ingestion.downloader import download_pdf
from utils.logging_config import get_logger

log = get_logger("pipeline")


def _new_stats() -> dict:
    return {"discovered": 0, "skipped_existing": 0, "downloaded": 0, "valid_new": 0,
            "invalid": 0, "needs_ocr": 0, "failed_downloads": 0, "metadata_only": 0,
            "duplicates": 0, "rejected_unverified": 0}


def _record_candidate(db: DatabaseManager, c: PaperCandidate, status: str, detail: str) -> None:
    db.insert_paper({
        "paper_id": c.paper_id, "title": c.title or "(untitled)", "abstract": c.abstract,
        "authors": c.authors, "publication_year": c.publication_year, "conference": c.conference,
        "doi": c.doi, "source_url": c.source_url, "pdf_url": c.pdf_url,
        "source_name": c.source_name, "venue_verification": c.venue_verification,
        "download_status": status, "status_detail": detail,
    })


def collect_conference(db: DatabaseManager, conference: str) -> dict:
    stats = _new_stats()
    target = target_for(conference)
    have = db.count_valid(conference)
    log.info("[%s] target=%d, already valid=%d", conference, target, have)
    adapters = adapters_for(conference)
    if not adapters:
        log.warning("[%s] no verified source configured; shortfall=%d", conference, target - have)

    for adapter in adapters:
        if have >= target:
            break
        log.info("[%s] source %s", conference, adapter.describe())
        for cand in adapter.candidates():
            if have >= target:
                break
            stats["discovered"] += 1
            existing = db.get_paper_by_source(cand.source_url)
            if existing and CONFIG["skip_existing_papers"] and existing["download_status"] in ("valid", "invalid", "needs_ocr", "duplicate"):
                stats["skipped_existing"] += 1
                continue

            if not cand.enriched:
                adapter.enrich(cand)
            if not cand.title:
                stats["failed_downloads"] += 1
                continue
            if not cand.pdf_url or not CONFIG["download_pdfs"]:
                if not existing:
                    _record_candidate(db, cand, "metadata_only", "no PDF URL")
                stats["metadata_only"] += 1
                continue
            if not existing:
                _record_candidate(db, cand, "pending", "")

            dest = conference_dir(conference) / f"{cand.paper_id}.pdf"
            check, msg = download_pdf(cand.pdf_url, dest)
            if check is None:
                db.update_paper(cand.paper_id, download_status="failed", status_detail=msg)
                stats["failed_downloads"] += 1
                continue
            stats["downloaded"] += 1
            if check.ok:
                dup = db.find_by_sha(check.sha256)
                if dup and dup["paper_id"] != cand.paper_id:
                    dest.unlink(missing_ok=True)
                    db.update_paper(cand.paper_id, download_status="duplicate",
                                    status_detail=f"same file as {dup['paper_id']}")
                    stats["duplicates"] += 1
                    continue
                db.update_paper(cand.paper_id, download_status="valid", status_detail=check.detail,
                                pdf_path=dest.relative_to(PROJECT_ROOT).as_posix(),
                                file_sha256=check.sha256, page_count=check.page_count,
                                full_text_available=1, title=cand.title, abstract=cand.abstract,
                                authors=cand.authors, doi=cand.doi)
                have += 1
                stats["valid_new"] += 1
                log.info("[%s] %d/%d  %s", conference, have, target, cand.title[:80])
            else:
                db.update_paper(cand.paper_id, download_status=check.status, status_detail=check.detail,
                                page_count=check.page_count)
                stats["needs_ocr" if check.status == "needs_ocr" else "invalid"] += 1
                log.warning("[%s] rejected %s: %s", conference, cand.source_url, check.detail)
        stats["rejected_unverified"] += adapter.stats["rejected_unverified"]

    stats["valid_total"] = have
    stats["target"] = target
    stats["shortfall"] = max(0, target - have)
    return stats


def collect(conferences: list[str] | None = None) -> dict:
    ensure_directories()
    db = DatabaseManager()
    for acr in CONFIG["conferences"]:
        db.upsert_conference(acr, CONFERENCE_INFO[acr])
    report = {"run_at": datetime.now(timezone.utc).isoformat(), "papers_per_conference": CONFIG["papers_per_conference"],
              "conferences": {}}
    for conf in conferences or active_conferences():
        report["conferences"][conf] = collect_conference(db, conf)
    out = path("reports_dir") / "collection_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    db.close()
    return report


def process(force: bool = False) -> dict:
    """Extract and chunk every valid reference PDF that is not yet processed."""
    from ingestion.chunker import chunk_document, chunk_signature
    from ingestion.extractor import ExtractionError, extract_pdf, load_extraction, save_extraction

    db = DatabaseManager()
    sig = chunk_signature()
    if db.get_setting("chunk_signature") not in (None, sig):
        log.warning("Chunking config changed (%s -> %s); re-chunking all papers", db.get_setting("chunk_signature"), sig)
        force = True
    stats = {"processed": 0, "skipped": 0, "failed": 0, "needs_ocr": 0, "chunks_added": 0}
    for conf in active_conferences():
        for p in db.list_papers(conf, "valid"):
            if p["extraction_status"] == "extracted" and not force:
                stats["skipped"] += 1
                continue
            try:
                doc = None if force else load_extraction(p["paper_id"])
                if doc is None:
                    doc = extract_pdf(PROJECT_ROOT / p["pdf_path"])
                    save_extraction(p["paper_id"], doc)
            except ExtractionError as exc:
                db.update_paper(p["paper_id"], extraction_status=exc.status, status_detail=str(exc))
                stats["needs_ocr" if exc.status == "needs_ocr" else "failed"] += 1
                log.warning("[%s] extraction %s: %s", p["paper_id"], exc.status, exc)
                continue
            chunks = chunk_document(doc, p["paper_id"], conf)
            if not chunks:
                db.update_paper(p["paper_id"], extraction_status="failed", status_detail="no chunks produced")
                stats["failed"] += 1
                continue
            old = [c["chunk_text"] for c in db.get_chunks(p["paper_id"])]
            if old != [c.chunk_text for c in chunks]:
                db.delete_chunks(p["paper_id"])  # cascades to vector_map; FAISS is revalidated on index
                stats["chunks_added"] += db.insert_chunks(c.as_row() for c in chunks)
            else:
                stats["chunks_unchanged_papers"] = stats.get("chunks_unchanged_papers", 0) + 1
            fields = {"extraction_status": "extracted"}
            if not p.get("abstract") and doc.abstract:
                fields["abstract"] = doc.abstract
            db.update_paper(p["paper_id"], **fields)
            stats["processed"] += 1
            log.info("[%s] %d sections, %d chunks", p["paper_id"], len(doc.sections), len(chunks))
    db.set_setting("chunk_signature", sig)
    db.close()
    return stats


def index(rebuild: bool = False, model_name: str | None = None) -> dict:
    from embeddings.embedding_generator import embed_pending

    db = DatabaseManager()
    try:
        return embed_pending(db, model_name, rebuild=rebuild)
    finally:
        db.close()


def status() -> list[dict]:
    db = DatabaseManager()
    rows = []
    for conf in CONFIG["conferences"]:
        counts = {r["download_status"]: r["n"] for r in db.status_counts() if r["conference"] == conf}
        on_disk = len(list(conference_dir(conf).glob("*.pdf"))) if conference_dir(conf).exists() else 0
        rows.append({"conference": conf, "target": target_for(conf), "valid": counts.get("valid", 0),
                     "pdfs_on_disk": on_disk, "discovered": sum(counts.values()),
                     "failed": counts.get("failed", 0), "invalid": counts.get("invalid", 0) + counts.get("needs_ocr", 0),
                     "metadata_only": counts.get("metadata_only", 0)})
    db.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--conference", choices=CONFIG["conferences"], action="append")
    p = sub.add_parser("process")
    p.add_argument("--force", action="store_true")
    i = sub.add_parser("index")
    i.add_argument("--rebuild", action="store_true")
    i.add_argument("--model")
    sub.add_parser("build")
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd in ("collect", "build"):
        report = collect(getattr(args, "conference", None))
        print(json.dumps(report["conferences"], indent=2))
    if args.cmd in ("process", "build"):
        print(json.dumps(process(getattr(args, "force", False)), indent=2))
    if args.cmd in ("index", "build"):
        print(json.dumps(index(getattr(args, "rebuild", False), getattr(args, "model", None)), indent=2))
    rows = status()
    print(f"\n{'conf':<8}{'target':>7}{'valid':>7}{'on_disk':>9}{'failed':>8}{'invalid':>9}{'meta_only':>11}")
    for r in rows:
        print(f"{r['conference']:<8}{r['target']:>7}{r['valid']:>7}{r['pdfs_on_disk']:>9}{r['failed']:>8}"
              f"{r['invalid']:>9}{r['metadata_only']:>11}")
    print(f"{'TOTAL':<8}{sum(r['target'] for r in rows):>7}{sum(r['valid'] for r in rows):>7}"
          f"{sum(r['pdfs_on_disk'] for r in rows):>9}")


if __name__ == "__main__":
    main()
