"""Conference rationale (<= max_rationale_words) and decision explanations."""
from __future__ import annotations

from pydantic import BaseModel

from config import CONFERENCE_INFO, CONFIG
from utils.llm import LLMUnavailable, complete_json, llm_available, model_tag
from utils.logging_config import get_logger
from utils.text_utils import truncate_words, word_count

log = get_logger("rationale")


class Rationale(BaseModel):
    rationale: str


PROMPT = """Write a formal justification (at most {max_words} words, one paragraph, no bullet points) for submitting
the paper to {venue}. Explain (1) how the paper's topic aligns with the venue, (2) how its methodology fits the
venue, (3) how its contribution relates to the venue's reference papers listed below (you may name at most two of
them by title), and (4) why the venue is suitable. Use only the information provided; do not invent venue policies,
deadlines, acceptance rates or citations.

<paper>
{paper}
</paper>

Venue profile (derived from its reference papers): {profile}
Most similar reference papers from {venue}:
{refs}"""


def conference_rationale(conference: str, profile_text: str, venue_profile: dict, match_entry: dict) -> tuple[str, str]:
    max_words = CONFIG["max_rationale_words"]
    refs = match_entry.get("top_reference_papers", [])[:3]
    venue = conference.upper()
    if llm_available():
        lp = venue_profile.get("llm_profile") or {}
        prof = lp.get("summary") or ", ".join(list(venue_profile["keywords"])[:12])
        ref_lines = "\n".join(f"- {r['title']} (similarity {r['similarity_score']}): {r['passage'][:300]}" for r in refs)
        try:
            out = complete_json(PROMPT.format(max_words=max_words - 10, venue=venue, paper=profile_text[:5000],
                                              profile=prof, refs=ref_lines or "(none)"), Rationale, max_tokens=1200)
            text = " ".join(out.rationale.split())
            if word_count(text) > max_words:
                text = truncate_words(text, max_words)
            return text, model_tag()
        except LLMUnavailable as exc:
            log.warning("Rationale fell back to template: %s", exc)
    # Template fallback built only from computed signals.
    terms = ", ".join(list(venue_profile["keywords"])[:4])
    ref = f' and is closest to the {venue} reference paper "{refs[0]["title"]}"' if refs else ""
    text = (f"The paper fits {venue} ({CONFERENCE_INFO[conference]}): its content aligns with themes prominent in "
            f"the venue's reference papers ({terms}){ref}. Its embedding-based similarity to the {venue} reference "
            f"corpus ranked highest among the candidate venues, indicating matching topic and methodology.")
    return truncate_words(text, max_words), "template-fallback"
