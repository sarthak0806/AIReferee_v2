"""Discovery of verified conference papers from official / permitted sources.

Each adapter lazily yields ``PaperCandidate`` objects in a deterministic order,
so increasing ``papers_per_conference`` simply continues further down the same
listings (and then on to the next configured source).
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import quote, urljoin

from bs4 import BeautifulSoup

from config import ARXIV_REJECT_PATTERNS, CONFERENCE_SOURCES
from utils.http import client
from utils.logging_config import get_logger

log = get_logger("collector")


@dataclass
class PaperCandidate:
    conference: str
    source_name: str
    source_url: str              # canonical landing page (unique key)
    title: str = ""
    pdf_url: str | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    venue_verification: str = ""
    enriched: bool = False

    @property
    def paper_id(self) -> str:
        digest = hashlib.sha1(self.source_url.encode("utf-8")).hexdigest()[:10]
        return f"{self.conference}_{self.publication_year or 'na'}_{digest}"


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _parse_citation_meta(html: str) -> tuple[dict[str, list[str]], BeautifulSoup]:
    soup = BeautifulSoup(html, "lxml")
    meta: dict[str, list[str]] = {}
    for tag in soup.find_all("meta"):
        name = tag.get("name") or ""
        if name.startswith("citation_") and tag.get("content"):
            meta.setdefault(name, []).append(tag["content"].strip())
    return meta, soup


def _flip_name(name: str) -> str:
    """'Zeng, Haijin' -> 'Haijin Zeng'."""
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        return f"{first} {last}".strip()
    return name.strip()


class SourceAdapter:
    source_name = "base"

    def __init__(self, conference: str, spec: dict) -> None:
        self.conference = conference
        self.spec = spec
        self.stats = {"rejected_unverified": 0}

    def candidates(self) -> Iterator[PaperCandidate]:
        raise NotImplementedError

    def enrich(self, cand: PaperCandidate) -> PaperCandidate:
        """Fetch landing-page metadata (abstract, authors, DOI, PDF URL)."""
        return cand

    def describe(self) -> str:
        return f"{self.source_name}:{self.spec}"


# --------------------------------------------------------------------------- CVF
class CVFAdapter(SourceAdapter):
    """CVF Open Access (official CVPR proceedings)."""
    source_name = "cvf_open_access"
    BASE = "https://openaccess.thecvf.com/"

    def candidates(self) -> Iterator[PaperCandidate]:
        event, year = self.spec["event"], self.spec["year"]
        html = client().get_text(f"{self.BASE}{event}?day=all")
        if not html:
            log.error("CVF listing for %s unavailable", event)
            return
        soup = BeautifulSoup(html, "lxml")
        for dt in soup.select("dt.ptitle"):
            a = dt.find("a")
            if not a or not a.get("href"):
                continue
            page = urljoin(self.BASE, a["href"])
            pdf = None
            for sib in dt.find_next_siblings():
                if sib.name == "dt":
                    break
                link = sib.find("a", string=re.compile(r"^\s*pdf\s*$"))
                if link:
                    pdf = urljoin(self.BASE, link["href"])
                    break
            yield PaperCandidate(
                conference=self.conference, source_name=self.source_name, source_url=page,
                title=_clean(a.get_text()) or "", pdf_url=pdf, publication_year=year,
                venue_verification=f"Listed in official CVF Open Access proceedings ({event})",
            )

    def enrich(self, cand: PaperCandidate) -> PaperCandidate:
        html = client().get_text(cand.source_url)
        if html:
            meta, soup = _parse_citation_meta(html)
            cand.authors = [_flip_name(n) for n in meta.get("citation_author", [])]
            cand.title = meta.get("citation_title", [cand.title])[0]
            abstract = soup.find(id="abstract")
            cand.abstract = _clean(abstract.get_text()) if abstract else None
            cand.pdf_url = meta.get("citation_pdf_url", [cand.pdf_url])[0] or cand.pdf_url
            cand.enriched = True
        return cand


# ----------------------------------------------------------------------- NeurIPS
def neurips_abstract(soup: BeautifulSoup) -> str | None:
    # The page nests <p> inside <p class="paper-abstract">, which parsers split
    # apart, so take the text of the whole "Abstract" section instead.
    heading = soup.find(["h2", "h4"], string=re.compile(r"^\s*Abstract\s*$"))
    if not heading:
        return None
    section = heading.find_parent("section") or heading.parent
    text = _clean(section.get_text(" "))
    return _clean(re.sub(r"^Abstract\s*", "", text or "")) or None


class NeurIPSAdapter(SourceAdapter):
    """Official NeurIPS proceedings (proceedings.neurips.cc), main track."""
    source_name = "neurips_proceedings"
    BASE = "https://proceedings.neurips.cc"

    def candidates(self) -> Iterator[PaperCandidate]:
        year = self.spec["year"]
        html = client().get_text(f"{self.BASE}/paper_files/paper/{year}")
        if not html:
            log.error("NeurIPS %s listing unavailable", year)
            return
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", attrs={"title": "paper title"}):
            href = a.get("href", "")
            # Main conference track only (exclude Datasets & Benchmarks track).
            if not href.endswith("-Abstract-Conference.html"):
                continue
            yield PaperCandidate(
                conference=self.conference, source_name=self.source_name,
                source_url=urljoin(self.BASE, href), title=_clean(a.get_text()) or "",
                publication_year=year,
                venue_verification=f"Listed in official NeurIPS {year} proceedings (main track)",
            )

    def enrich(self, cand: PaperCandidate) -> PaperCandidate:
        html = client().get_text(cand.source_url)
        if html:
            meta, soup = _parse_citation_meta(html)
            cand.title = meta.get("citation_title", [cand.title])[0]
            cand.authors = [_flip_name(n) for n in meta.get("citation_author", [])]
            cand.doi = (meta.get("citation_doi") or [None])[0]
            cand.pdf_url = (meta.get("citation_pdf_url") or [None])[0]
            cand.abstract = neurips_abstract(soup)
            cand.enriched = True
        return cand


# --------------------------------------------------------------- ACL Anthology
class ACLAnthologyAdapter(SourceAdapter):
    """ACL Anthology volumes (official EMNLP proceedings)."""
    source_name = "acl_anthology"
    BASE = "https://aclanthology.org"
    MAX_CONSECUTIVE_MISSES = 3

    def candidates(self) -> Iterator[PaperCandidate]:
        volume, year = self.spec["volume"], self.spec["year"]
        n, misses = 1, 0  # .0 is front matter
        while misses < self.MAX_CONSECUTIVE_MISSES:
            yield PaperCandidate(
                conference=self.conference, source_name=self.source_name,
                source_url=f"{self.BASE}/{volume}.{n}/", pdf_url=f"{self.BASE}/{volume}.{n}.pdf",
                publication_year=year,
                venue_verification=f"Published in ACL Anthology volume {volume}",
            )
            n += 1
            # misses are tracked via enrich(); see _missed flag
            if getattr(self, "_missed", False):
                misses += 1
                self._missed = False
            else:
                misses = 0

    def enrich(self, cand: PaperCandidate) -> PaperCandidate:
        html = client().get_text(cand.source_url)
        if not html:
            self._missed = True
            return cand
        meta, soup = _parse_citation_meta(html)
        cand.title = (meta.get("citation_title") or [""])[0]
        cand.authors = meta.get("citation_author", [])
        cand.doi = (meta.get("citation_doi") or [None])[0]
        cand.pdf_url = (meta.get("citation_pdf_url") or [cand.pdf_url])[0]
        abstract = soup.select_one(".acl-abstract span")
        cand.abstract = _clean(abstract.get_text()) if abstract else None
        cand.enriched = bool(cand.title)
        return cand


# ------------------------------------------------------------------------ arXiv
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def venue_year(evidence: str, updated: str = "") -> int | None:
    """Venue year from the acceptance statement (e.g. 'TMLR (12/2023)'), else the
    latest arXiv version date. The first-submission date is not the venue year."""
    years = [int(y) for y in re.findall(r"\b(20[12]\d)\b", evidence or "")]
    if years:
        return max(years)
    return int(updated[:4]) if updated[:4].isdigit() else None


class ArxivVenueAdapter(SourceAdapter):
    """arXiv papers whose *author-supplied* comment or journal_ref states that the
    paper was accepted at / published in the venue. Topic similarity is never used."""
    source_name = "arxiv"
    API = "https://export.arxiv.org/api/query"
    PAGE = 100

    def __init__(self, conference: str, spec: dict) -> None:
        super().__init__(conference, spec)
        self.accept = [re.compile(p, re.I) for p in spec["accept_patterns"]]
        self.reject = [re.compile(p, re.I) for p in ARXIV_REJECT_PATTERNS]

    def verify(self, comment: str, journal_ref: str) -> str | None:
        for field_name, text in (("journal_ref", journal_ref), ("comment", comment)):
            if not text:
                continue
            if any(r.search(text) for r in self.reject):
                continue
            if any(a.search(text) for a in self.accept):
                return f'arXiv {field_name} states venue acceptance: "{text[:200]}"'
        return None

    def candidates(self) -> Iterator[PaperCandidate]:
        start = 0
        while True:
            url = (f"{self.API}?search_query={quote(self.spec['query'])}&start={start}"
                   f"&max_results={self.PAGE}&sortBy=submittedDate&sortOrder=ascending")
            xml = client().get_text(url)
            if not xml:
                log.error("arXiv query failed at offset %d", start)
                return
            root = ET.fromstring(xml)
            entries = root.findall("a:entry", ATOM)
            if not entries:
                return
            for e in entries:
                comment = _clean(e.findtext("arxiv:comment", "", ATOM)) or ""
                jref = _clean(e.findtext("arxiv:journal_ref", "", ATOM)) or ""
                evidence = self.verify(comment, jref)
                if not evidence:
                    self.stats["rejected_unverified"] += 1
                    continue
                abs_url = e.findtext("a:id", "", ATOM).strip()
                arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])
                pdf = next((l.get("href") for l in e.findall("a:link", ATOM) if l.get("title") == "pdf"),
                           f"https://arxiv.org/pdf/{arxiv_id}")
                updated = e.findtext("a:updated", "", ATOM)
                yield PaperCandidate(
                    conference=self.conference, source_name=self.source_name,
                    source_url=f"https://arxiv.org/abs/{arxiv_id}",
                    title=_clean(e.findtext("a:title", "", ATOM)) or "",
                    pdf_url=pdf,
                    authors=[_clean(a.findtext("a:name", "", ATOM)) for a in e.findall("a:author", ATOM)],
                    abstract=_clean(e.findtext("a:summary", "", ATOM)),
                    publication_year=self.spec.get("year") or venue_year(evidence, updated),
                    doi=_clean(e.findtext("arxiv:doi", "", ATOM)),
                    venue_verification=evidence, enriched=True,
                )
            start += self.PAGE
            total = int(root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults", "0"))
            if start >= total:
                return


ADAPTERS = {
    "cvf": CVFAdapter,
    "neurips": NeurIPSAdapter,
    "acl": ACLAnthologyAdapter,
    "arxiv": ArxivVenueAdapter,
}


def adapters_for(conference: str) -> list[SourceAdapter]:
    return [ADAPTERS[spec["type"]](conference, spec) for spec in CONFERENCE_SOURCES.get(conference, [])]
