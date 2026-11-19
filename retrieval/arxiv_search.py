"""arXiv API search (metadata + abstracts only; results are not full-text evidence)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

from config import CONFIG
from database.db_manager import DatabaseManager
from utils.http import client
from utils.logging_config import get_logger

log = get_logger("arxiv_search")

ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
API = "https://export.arxiv.org/api/query"
_STOP = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "with", "to", "using", "via", "based", "by"}


def _to_arxiv_query(text: str) -> str:
    """Turn a natural-language query into an arXiv boolean query over title+abstract."""
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", text.lower()) if w not in _STOP and len(w) > 2][:8]
    return " AND ".join(f"all:{w}" for w in words) if words else f'all:"{text}"'


def parse_feed(xml: str) -> list[dict]:
    root = ET.fromstring(xml)
    out = []
    for e in root.findall("a:entry", ATOM):
        abs_url = (e.findtext("a:id", "", ATOM) or "").strip()
        if "/abs/" not in abs_url:
            continue
        arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])
        pdf = next((l.get("href") for l in e.findall("a:link", ATOM) if l.get("title") == "pdf"), None)
        published = e.findtext("a:published", "", ATOM)
        out.append({
            "source": "arxiv",
            "title": re.sub(r"\s+", " ", e.findtext("a:title", "", ATOM)).strip(),
            "authors": [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)],
            "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", "", ATOM)).strip(),
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "published": published[:10],
            "arxiv_id": arxiv_id,
            "doi": e.findtext("arxiv:doi", None, ATOM),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf,
            "content_type": "metadata_only",
        })
    return out


def search_arxiv(query: str, db: DatabaseManager | None = None, max_results: int | None = None) -> list[dict]:
    if not CONFIG["enable_arxiv"]:
        return []
    n = max_results or CONFIG["external_results_per_query"]
    key = f"{query}|{n}"
    if db and CONFIG["external_search_cache"]:
        cached = db.get_cached_search("arxiv", key)
        if cached is not None:
            return cached
    url = f"{API}?search_query={quote(_to_arxiv_query(query))}&start=0&max_results={n}&sortBy=relevance"
    xml = client().get_text(url)
    if xml is None:
        log.warning("arXiv search failed for %r", query)
        return []
    try:
        results = parse_feed(xml)
    except ET.ParseError as exc:
        log.warning("arXiv returned malformed XML: %s", exc)
        return []
    for r in results:
        r["query"] = query
    if db and CONFIG["external_search_cache"]:
        db.cache_search("arxiv", key, results)
    return results
