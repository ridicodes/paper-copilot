"""Conservative sentence-level citation completeness checks.

This checks citation syntax and coverage, not whether a source entails a claim.
Unrecognized prose formatting falls back to the app's extractive answer.
"""
import re


def citations_are_complete(answer: str) -> bool:
    text = answer.replace("**", "").replace("__", "")
    # Treat citations after a sentence's punctuation as belonging to it.
    text = re.sub(r"([.!?])\s*((?:\[E\d+\][ \t]*)+)", r" \2\1", text,
                  flags=re.IGNORECASE)
    units = re.split(r"\n+|(?<=[.!?])\s+(?=[A-Za-z0-9*•\-])", text)
    checked = 0
    for unit in units:
        unit = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", unit).strip()
        if not unit:
            continue
        # Require even headings/introductions to be omitted or cited. This avoids
        # mistaking an uncited factual conclusion formatted as a heading for a label.
        if not re.search(r"\[E\d+\](?:\s*\[E\d+\])*[.!?]?\s*$", unit, re.I):
            return False
        if not re.search(r"[A-Za-z]", re.sub(r"\[E\d+\]", "", unit, flags=re.I)):
            return False
        checked += 1
    return checked > 0


def normalize_answer(answer: str, query: str) -> str:
    """Remove a repeated question or an introductory label made from its words."""
    lines = answer.strip().splitlines()
    words = set(re.findall(r"[a-z]+", query.lower())) | {
        "the", "following", "are", "is", "in", "ways", "performs", "two", "on"
    }
    while lines:
        first = lines[0].strip()
        label = first.rstrip(":")
        if not first or first.casefold() == query.strip().casefold() or (
            first.endswith(":") and set(re.findall(r"[a-z]+", label.lower())) <= words
        ):
            lines.pop(0)
        else:
            break
    return "\n".join(lines).strip()
