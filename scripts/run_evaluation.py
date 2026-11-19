"""Batch-evaluate the hackathon dataset and write results.csv.

Place the hackathon PDFs in data/evaluation_dataset/papers/ (file stem = Paper ID,
e.g. P001.pdf). Optional ground truth: data/evaluation_dataset/labels.csv with
columns "Paper ID,Publishable[,Conference]". Labelled papers are evaluated like
any other paper: labels are only read afterwards to compute metrics, and are
never shown to the pipeline (no leakage).

Usage (from AIReferee/):
    python -m scripts.run_evaluation                    # all PDFs
    python -m scripts.run_evaluation --only P001 P002   # subset
    python -m scripts.run_evaluation --labelled-only    # just the labelled reference papers
    python -m scripts.run_evaluation --no-external      # skip arXiv/Scholar (saves SerpAPI quota)
"""
from __future__ import annotations

import argparse
import csv
import json
import traceback

from config import path
from database.db_manager import DatabaseManager
from evaluation.final_evaluator import csv_row, evaluate_paper
from evaluation.metrics import classification_metrics
from reporting.csv_generator import write_results_csv
from reporting.pdf_report import build_pdf
from utils.logging_config import get_logger

log = get_logger("run_evaluation")


def load_labels(p) -> dict[str, dict]:
    if not p.exists():
        return {}
    with p.open(encoding="utf-8-sig") as f:
        return {r["Paper ID"].strip(): r for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--labelled-only", action="store_true")
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="reuse existing result.json files")
    args = ap.parse_args()

    base = path("evaluation_dataset_dir")
    pdfs = sorted((base / "papers").glob("*.pdf"))
    labels = load_labels(base / "labels.csv")
    if args.only:
        pdfs = [p for p in pdfs if p.stem in set(args.only)]
    if args.labelled_only:
        pdfs = [p for p in pdfs if p.stem in labels]
    if not pdfs:
        print(f"No PDFs found in {base / 'papers'}")
        return

    db = DatabaseManager()
    rows, results, failures = [], [], []
    for i, pdf in enumerate(pdfs, 1):
        out_json = path("reports_dir") / pdf.stem / "result.json"
        try:
            if args.skip_done and out_json.exists():
                res = json.loads(out_json.read_text(encoding="utf-8"))
            else:
                log.info("[%d/%d] %s", i, len(pdfs), pdf.name)
                res = evaluate_paper(pdf, paper_id=pdf.stem, db=db, use_external=not args.no_external)
                (out_json.parent / "report.pdf").write_bytes(build_pdf(res))
            results.append(res)
            rows.append(csv_row(res))
        except Exception as exc:
            log.error("%s failed: %s", pdf.name, exc)
            failures.append({"paper": pdf.name, "error": str(exc), "trace": traceback.format_exc()[-1500:]})
            # Policy for papers that cannot be processed: conservative 0 / na, recorded in failures.json
            rows.append({"Paper ID": pdf.stem, "Publishable": 0, "Conference": "na", "Rationale": "na"})

    out_csv = write_results_csv(rows, path("reports_dir") / "results.csv")
    summary = {"papers": len(pdfs), "failures": len(failures), "results_csv": str(out_csv),
               "decisions": {}}
    for r in results:
        d = r["decision"]["decision"]
        summary["decisions"][d] = summary["decisions"].get(d, 0) + 1

    labelled = [r for r in rows if r["Paper ID"] in labels]
    if labelled:
        y_true = [int(labels[r["Paper ID"]]["Publishable"]) for r in labelled]
        y_pred = [int(r["Publishable"]) for r in labelled]
        summary["classification_metrics"] = classification_metrics(y_true, y_pred)
        conf_pairs = [(labels[r["Paper ID"]].get("Conference", "").strip().lower(), r["Conference"])
                      for r in labelled if labels[r["Paper ID"]].get("Conference", "").strip().lower()
                      not in ("", "na") and r["Conference"] != "na"]
        if conf_pairs:
            summary["conference_accuracy_on_labelled_publishable"] = {
                "n": len(conf_pairs), "accuracy": round(sum(a == b for a, b in conf_pairs) / len(conf_pairs), 4)}
    (path("reports_dir") / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        (path("reports_dir") / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
