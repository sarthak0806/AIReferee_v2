"""Detailed per-paper evaluation report (ReportLab)."""
from __future__ import annotations

import io
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from evaluation.publishability_classifier import SPECS
from evaluation.rubric import DIMENSIONS

NAVY = colors.HexColor("#1b2f4b")
GRID = colors.HexColor("#c9d1dc")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=8.5, leading=11))
    ss.add(ParagraphStyle("Cell", parent=ss["BodyText"], fontSize=8, leading=10))
    ss.add(ParagraphStyle("H", parent=ss["Heading2"], textColor=NAVY, spaceBefore=8, spaceAfter=4))
    return ss


def _p(text, style) -> Paragraph:
    return Paragraph(escape(str(text if text not in (None, "") else "—")).replace("\n", "<br/>"), style)


def _table(rows, widths, ss, header=True) -> Table:
    data = [[_p(c, ss["Cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def _bullets(items, ss) -> list:
    return [_p(f"• {i}", ss["Small"]) for i in items] if items else [_p("None recorded.", ss["Small"])]


def build_pdf(result: dict) -> bytes:
    ss = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=f"AIReferee report {result['paper_id']}")
    W = A4[0] - 32 * mm
    a = result["analysis"]
    prof = a["profile"]
    d = result["decision"]
    asmt = result["assessment"]
    story = [Paragraph("AIReferee Evaluation Report", ss["Title"])]

    # 1. Paper details
    story += [Paragraph("1. Paper details", ss["H"]),
              _table([["Field", "Value"], ["Paper ID", result["paper_id"]], ["File", result["pdf_name"]],
                      ["Title", prof["title"]], ["Pages", a["page_count"]],
                      ["Sections detected", ", ".join(a["sections_found"])],
                      ["Profile source", a["profile_source"]], ["Evaluated", result["evaluated_at"]]],
                     [35 * mm, W - 35 * mm], ss)]
    # 2-3. Abstract and contribution
    story += [Paragraph("2. Abstract", ss["H"]), _p(prof["abstract"], ss["Small"]),
              Paragraph("3. Research contribution", ss["H"]), _p(prof["research_problem"], ss["Small"]),
              *_bullets(prof["claimed_contributions"], ss)]
    # 4. Classification
    story += [Paragraph("4. Publishability classification", ss["H"]),
              _table([["Decision", "Weighted score", "Confidence", "CSV label", "Assessment source"],
                      [d["decision"], d["weighted_score"], d["confidence"], d["csv_publishable"],
                       d["assessment_source"]]], [W / 5] * 5, ss),
              Spacer(1, 4), *_bullets(d["reasons"] + d["blocking_issues"], ss)]
    # 5. Dimensions
    rows = [["Dimension", "Score", "Conf.", "Finding"]]
    for dim in DIMENSIONS:
        x = asmt[dim]
        rows.append([SPECS[dim].label, x["score"], x["confidence"], x["finding"]])
    story += [Paragraph("5. Dimension-wise evaluation", ss["H"]),
              _table(rows, [35 * mm, 14 * mm, 14 * mm, W - 63 * mm], ss)]
    for dim in DIMENSIONS:
        x = asmt[dim]
        story += [Paragraph(f"<b>{SPECS[dim].label}</b>", ss["Small"]),
                  _p("Strengths: " + "; ".join(x["strengths"] or ["—"]), ss["Small"]),
                  _p("Weaknesses: " + "; ".join(x["weaknesses"] or ["—"]), ss["Small"]),
                  _p("Supporting evidence: " + ", ".join(x["supporting_evidence"] or ["—"]) +
                     " | Counter-evidence: " + ", ".join(x["counter_evidence"] or ["—"]), ss["Small"]),
                  _p("Missing evidence: " + "; ".join(x["missing_evidence"] or ["—"]), ss["Small"]),
                  Spacer(1, 3)]
    story.append(PageBreak())
    # 6-7. Retrieved papers and evidence
    story += [Paragraph("6. Selected evidence (full-text passages)", ss["H"])]
    rows = [["ID", "Paper / location", "Bi-enc.", "Rerank", "Relevance"]]
    for e in result["evidence"]:
        rows.append([e["evidence_id"], f"{e['title']} — {str(e.get('conference') or '').upper()} {e.get('year') or ''}, "
                     f"{e.get('section')}, p.{e.get('page')}\n{e.get('source_url') or ''}",
                     e["similarity_score"], e["reranker_score"], e["relevance"]])
    story.append(_table(rows, [10 * mm, W - 72 * mm, 18 * mm, 18 * mm, 26 * mm], ss))
    for e in result["evidence"]:
        story += [Spacer(1, 3), _p(f"[{e['evidence_id']}] “{e['passage'][:900]}”", ss["Small"])]
    story += [Paragraph("7. Related external literature (metadata only)", ss["H"])]
    rows = [["ID", "Title", "Year", "Source", "Rerank"]]
    for x in result["external_results"][:12]:
        rows.append([x["evidence_id"], f"{x['title']}\n{x.get('url') or ''}", x.get("year"),
                     ", ".join(s for s in x.get("sources", []) if s), x.get("reranker_score")])
    story.append(_table(rows, [10 * mm, W - 66 * mm, 12 * mm, 26 * mm, 18 * mm], ss))
    # 8. Decision rationale
    story += [Paragraph("8. Decision rationale", ss["H"]), _p(asmt.get("overall_summary"), ss["Small"])]
    if asmt.get("unsubstantiated_claims"):
        story += [_p("Unsubstantiated claims:", ss["Small"]), *_bullets(asmt["unsubstantiated_claims"], ss)]
    if asmt.get("coherence_issues"):
        story += [_p("Coherence issues:", ss["Small"]), *_bullets(asmt["coherence_issues"], ss)]
    # 9-10. Conference
    story += [Paragraph("9. Conference recommendation", ss["H"])]
    conf = result.get("conference")
    if conf:
        rows = [["Rank", "Venue", "Score", "Signals"]]
        for i, r in enumerate(conf["ranking"], 1):
            rows.append([i, r["conference"].upper(), r["score"],
                         ", ".join(f"{k}={v}" for k, v in r["signals"].items())])
        story += [_table(rows, [12 * mm, 22 * mm, 18 * mm, W - 52 * mm], ss),
                  Paragraph("10. Conference rationale", ss["H"]),
                  _p(f"{conf['recommended'].upper()}: {conf['rationale']}", ss["Small"]),
                  _p(f"(source: {conf['rationale_source']})", ss["Small"])]
    else:
        story += [_p("Not applicable: the paper was not classified as publishable.", ss["Small"])]
    # 11. Revision roadmap
    story += [Paragraph("11. Revision recommendations", ss["H"]), *_bullets(asmt.get("revision_roadmap"), ss)]
    # 12. Limitations, confidence and audit trail
    story += [Paragraph("12. Limitations, confidence and audit trail", ss["H"]),
              _p("This is an automated, AI-assisted assessment. Weights and thresholds are configurable starting "
                 "assumptions, not validated publication standards; similarity is not evidence of plagiarism; "
                 "external results are metadata only. A human reviewer should confirm the decision.", ss["Small"]),
              _p(f"Overall confidence: {d['confidence']}", ss["Small"]),
              _table([["Category", "Generated query"]] + [[q["category"], q["query"]] for q in result["queries"]],
                     [35 * mm, W - 35 * mm], ss),
              Spacer(1, 4),
              _table([["Setting", "Value"]] + [[k, str(v)] for k, v in result["config"].items()] +
                     [["timings (s)", str(result["timings_s"])]], [45 * mm, W - 45 * mm], ss)]
    doc.build(story)
    return buf.getvalue()
