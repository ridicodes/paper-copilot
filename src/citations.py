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


def repair_comparison_synthesis_citations(
    answer: str,
    evidence_map: dict[str, dict],
) -> str:
    """Cite one uncited A/B contrast with evidence already cited for both papers.

    This does not relax ``citations_are_complete``. It only repairs the common
    comparison shape where cited paper-specific bullets are followed by one
    uncited sentence explicitly contrasting those same paper labels.
    """
    used_ids = [match.upper() for match in re.findall(r"\[(E\d+)\]", answer, re.I)]
    if not used_ids or any(evidence_id not in evidence_map for evidence_id in used_ids):
        return answer

    cited_by_document: dict[str, str] = {}
    for evidence_id in used_ids:
        document = str(evidence_map[evidence_id].get("document", ""))
        if document:
            cited_by_document.setdefault(document, evidence_id)
    if len(cited_by_document) < 2:
        return answer

    repaired_lines: list[str] = []
    repairs = 0
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or re.search(r"\[E\d+\]", stripped, re.I):
            repaired_lines.append(line)
            continue

        paper_labels = set(re.findall(r"\bPaper\s+([A-Z0-9]+)\b", stripped, re.I))
        contrast = bool(re.search(
            r"\b(?:differ\w*|whereas|while|in contrast|compared with|compared to)\b",
            stripped,
            re.I,
        ))
        sentence_count = len([
            unit for unit in re.split(r"(?<=[.!?])\s+", stripped) if unit.strip()
        ])
        if repairs == 0 and contrast and len(paper_labels) >= 2 and sentence_count == 1:
            citations = " ".join(
                f"[{evidence_id}]" for evidence_id in cited_by_document.values()
            )
            repaired_lines.append(f"{line.rstrip()} {citations}")
            repairs += 1
        else:
            repaired_lines.append(line)

    return "\n".join(repaired_lines).strip()


def comparison_citations_cover_documents(
    answer: str,
    evidence_map: dict[str, dict],
) -> bool:
    """Require a comparison answer to cite evidence from every selected paper."""
    required_documents = {
        str(evidence.get("document", "")) for evidence in evidence_map.values()
        if evidence.get("document")
    }
    cited_documents = {
        str(evidence_map[evidence_id].get("document", ""))
        for evidence_id in (
            match.upper() for match in re.findall(r"\[(E\d+)\]", answer, re.I)
        )
        if evidence_id in evidence_map
    }
    return len(required_documents) >= 2 and required_documents <= cited_documents
