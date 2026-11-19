"""Heading detection and canonical section mapping."""
from __future__ import annotations

import re

CANONICAL_SECTIONS = [
    "Abstract", "Introduction", "Related Work", "Methodology", "Experiments", "Results",
    "Discussion", "Conclusion", "Limitations", "References", "Appendix",
]

# Ordered: first match wins.
_SECTION_RULES: list[tuple[str, re.Pattern]] = [
    ("Abstract", re.compile(r"^abstract\b", re.I)),
    ("Introduction", re.compile(r"^(introduction|motivation|overview)\b", re.I)),
    ("Related Work", re.compile(r"(related\s*work|background|prior\s*work|literature|preliminar)", re.I)),
    ("References", re.compile(r"^(references|bibliography|works\s*cited)\b", re.I)),
    ("Appendix", re.compile(r"^(appendix|appendices|supplementary|supplemental)\b", re.I)),
    ("Limitations", re.compile(r"^(limitations?|broader\s*impact|ethic|societal)", re.I)),
    ("Conclusion", re.compile(r"(conclu|summary|future\s*work|closing)", re.I)),
    ("Discussion", re.compile(r"^discussion", re.I)),
    ("Results", re.compile(r"(results?|analysis|ablation|findings|case\s*stud|comparison)", re.I)),
    ("Experiments", re.compile(r"(experiment|evaluation|setup|implementation|dataset|benchmark|empirical)", re.I)),
    ("Methodology", re.compile(r"(method|approach|model|framework|proposed|architecture|algorithm|problem|formulation|theor|analysis\s*of|design|our\s)", re.I)),
]

# "3", "3.", "3.1", "III.", "A.", "A.1" followed by a capitalised word.
# Letters need a trailing dot or sub-number so a title like "A Simple Baseline" is not a heading.
_NUMBERED = re.compile(r"^(?P<num>(\d{1,2}(\.\d{1,2}){0,3}\.?)|([IVX]{1,5}\.)|([A-H]\.)|([A-H](\.\d{1,2})+\.?))"
                       r"\s+(?P<title>[A-Z][^.]{1,90})$")
_BARE_TOP = re.compile(r"^(abstract|introduction|related ?work|background|conclusions?|references|acknowledge?ments?|"
                       r"discussion|limitations?|appendix|bibliography|experiments?|results|methods?|methodology)$", re.I)


def canonical_section(heading: str) -> str | None:
    h = re.sub(r"^[\dIVXA-H.]+\s+", "", heading.strip())
    for name, pat in _SECTION_RULES:
        if pat.search(h):
            return name
    return None


def parse_heading(line: str, is_emphasised: bool) -> tuple[str, int] | None:
    """Return (heading_text, level) if ``line`` looks like a section heading.

    level 1 = top-level section, 2+ = subsection. Requires typographic
    emphasis (bold / larger font) unless the line is an exact known name.
    """
    text = line.strip()
    if not text or len(text) > 90 or text.endswith((",", ";")):
        return None
    # Some fonts drop inter-word spaces in headings ("RelatedWork"): restore them.
    if is_emphasised and " " not in text.split(" ", 1)[-1] and re.search(r"[a-z][A-Z]", text):
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    if _BARE_TOP.match(text):
        return text, 1
    if not is_emphasised:
        return None
    m = _NUMBERED.match(text)
    if m:
        num = m.group("num").rstrip(".")
        level = 1 if re.fullmatch(r"\d{1,2}|[IVX]{1,5}|[A-H]", num) else num.count(".") + 1
        return text, level
    return None


def is_acknowledgement(heading: str) -> bool:
    return bool(re.search(r"acknowledge?ment", heading, re.I))
