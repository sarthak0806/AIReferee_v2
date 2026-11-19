"""Google Scholar discovery through SerpAPI (an approved provider).

Results are metadata/snippets only and are never used as full-text evidence.
The free SerpAPI plan has 250 searches per month, so the number of Scholar
queries per paper is capped and responses are cached in SQLite.
"""
from __future__ import annotations

import re

import requests

from config import CONFIG, SERPAPI_API_KEY
from database.db_manager import DatabaseManager
from utils.logging_config import get_logger

log = get_logger("scholar_search")

ENDPOINT = "https://serpapi.com/search.json"


def scholar_available() -> bool:
    return bool(CONFIG["enable_google_scholar"] and SERPAPI_API_KEY)


def parse_results(payload: dict, query: str) -> list[dict]:
    out = []
    for r in payload.get("organic_results", []):
        info = r.get("publication_info", {}) or {}
        summary = info.get("summary", "") or ""
        year_match = re.search(r"\b((?:19|20)\d{2})\b", summary)
        authors = [a.get("name") for a in info.get("authors", []) if a.get("name")]
        if not authors and " - " in summary:
            authors = [a.strip() for a in summary.split(" - ")[0].split(",") if a.strip() and "…" not in a]
        link = r.get("link") or ""
        doi = None
        m = re.search(r"(10\.\d{4,9}/[^\s?#]+)", link)
        if m:
            doi = m.group(1)
        arxiv = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", link)
        pdf = next((res.get("link") for res in r.get("resources", []) or [] if res.get("file_format") == "PDF"), None)
        out.append({
            "source": "google_scholar",
            "title": r.get("title", "").strip(),
            "authors": authors,
            "snippet": r.get("snippet", ""),
            "abstract": None,
            "year": int(year_match.group(1)) if year_match else None,
            "publication_summary": summary,
            "doi": doi,
            "arxiv_id": arxiv.group(1) if arxiv else None,
            "url": link,
            "pdf_url": pdf,
            "cited_by": ((r.get("inline_links") or {}).get("cited_by") or {}).get("total"),
            "content_type": "metadata_only",
            "query": query,
        })
    return out


def search_scholar(query: str, db: DatabaseManager | None = None, num: int | None = None) -> list[dict]:
    if not scholar_available():
        return []
    n = num or CONFIG["external_results_per_query"]
    key = f"{query}|{n}"
    if db and CONFIG["external_search_cache"]:
        cached = db.get_cached_search("google_scholar", key)
        if cached is not None:
            return cached
    try:
        resp = requests.get(ENDPOINT, params={"engine": "google_scholar", "q": query, "num": n,
                                              "api_key": SERPAPI_API_KEY}, timeout=CONFIG["http_timeout_s"])
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("SerpAPI request failed: %s", exc)
        return []
    if resp.status_code != 200 or payload.get("error"):
        log.warning("SerpAPI error %s: %s", resp.status_code, payload.get("error"))
        return []
    results = parse_results(payload, query)
    if db and CONFIG["external_search_cache"]:
        db.cache_search("google_scholar", key, results)
    return results
