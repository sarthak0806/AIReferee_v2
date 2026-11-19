"""Hackathon results.csv writer with format validation."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from config import CONFIG
from utils.text_utils import word_count

COLUMNS = ["Paper ID", "Publishable", "Conference", "Rationale"]


def validate_row(row: dict) -> list[str]:
    errs = []
    if row["Publishable"] not in (0, 1):
        errs.append("Publishable must be 0 or 1")
    if row["Publishable"] == 0 and (row["Conference"] != "na" or row["Rationale"] != "na"):
        errs.append("Non-publishable rows need Conference=na and Rationale=na")
    if row["Publishable"] == 1:
        if row["Conference"] not in CONFIG["conferences"]:
            errs.append(f"Conference must be one of {CONFIG['conferences']}")
        if word_count(row["Rationale"]) > CONFIG["max_rationale_words"]:
            errs.append("Rationale exceeds the word limit")
    return errs


def to_csv_string(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writeheader()
    for r in rows:
        errs = validate_row(r)
        if errs:
            raise ValueError(f"{r['Paper ID']}: {'; '.join(errs)}")
        w.writerow({k: r[k] for k in COLUMNS})
    return buf.getvalue()


def write_results_csv(rows: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_csv_string(rows), encoding="utf-8")
    return out_path
