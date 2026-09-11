"""Conservative retrieval gates; similarity is relevance, not proof of an answer."""

import re
import unicodedata

from src.index import (
    GENERIC_QUERY_WORDS,
    STOPWORDS,
    comparison_requests_methods,
    is_methodology_question,
    methodology_score,
    method_purpose_score,
    query_terms,
    tokenize,
)

LEXICAL_THRESHOLD = 0.50
SEMANTIC_THRESHOLD = 0.55
METHODOLOGY_SEMANTIC_THRESHOLD = 0.35
CURRENCY_WORDS = {"dollar", "dollars", "usd", "euro", "euros", "rupee", "rupees"}
COMPARISON_META_TERMS = {
    "achieve", "achieved", "achieves", "address", "addresses", "aim", "aims",
    "approach", "approaches",
    "both", "challenge", "challenges", "compare", "compared", "comparison",
    "contrast", "differ", "difference", "differences", "differently", "does", "each",
    "goal", "goals", "method", "methods", "objective", "objectives", "paper",
    "papers", "problem",
    "problems", "purpose", "purposes", "role", "roles", "similar",
    "serve", "serves", "similarities", "similarity", "solve", "solves",
    "technique", "techniques", "their", "try", "two", "use", "uses",
    "central", "contribute", "contributes", "contributing", "solving", "solution",
}


def comparison_topic_terms(query: str) -> set[str]:
    """Return subject terms, excluding words that only express comparison intent."""
    return set(query_terms(query)) - COMPARISON_META_TERMS


def passage_is_relevant(result: dict) -> bool:
    if float(result.get("noise_penalty", 0)) >= 3.0:
        return False
    lexical = (float(result.get("coverage", 0)) >= LEXICAL_THRESHOLD
               and float(result.get("bm25_score", result.get("score", 0))) > 0)
    # Prototype threshold checked against the local library; not a probability.
    semantic = (result.get("retrieval_method") in {"semantic", "hybrid"}
                and float(result.get("semantic_score", 0)) >= SEMANTIC_THRESHOLD)
    return lexical or semantic


def query_anchors(query: str) -> set[str]:
    """Keep acronyms, versioned identifiers, and names within the question."""
    words = re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", query)
    anchors = {word.lower() for position, word in enumerate(words)
            if (len(word) >= 2 and word.isupper() and word.isalpha())
            or (any(c.isalpha() for c in word) and any(c.isdigit() for c in word))
            or (position > 0 and word[0].isupper()
                and word.lower() not in STOPWORDS | GENERIC_QUERY_WORDS)}
    anchors.update(word.lower() for word in words if word.lower() in CURRENCY_WORDS)
    return anchors


def supported_passages(query: str, results: list[dict]) -> list[dict]:
    anchors = query_anchors(query)
    strong_documents = {r["document"] for r in results if passage_is_relevant(r)
                        and anchors.issubset(set(tokenize(r.get("text", ""), False)))}
    return [result for result in results if (passage_is_relevant(result) or (
        is_methodology_question(query) and result["document"] in strong_documents
        and float(result.get("semantic_score", 0)) >= METHODOLOGY_SEMANTIC_THRESHOLD
        and float(result.get("noise_penalty", 0)) < 3.0
        and methodology_score(result.get("text", "")) >= 2))
            and anchors.issubset(set(tokenize(
                unicodedata.normalize("NFKC", result.get("text", "")),
                remove_stopwords=False,
            )))]


def evidence_is_sufficient(query: str, results: list[dict], comparison=False) -> bool:
    """Return whether the retrieved evidence is strong enough to answer safely.

    For cross-paper method/technique questions, support is aggregated at the
    document level. A paper may name a method in one passage and explain its
    purpose in another; forcing both facets into one chunk caused valid
    comparison questions to be rejected.
    """
    supported = supported_passages(query, results)

    if not comparison:
        return bool(supported)

    anchors = query_anchors(query)
    topics = comparison_topic_terms(query)
    method_question = comparison_requests_methods(query)

    # --------------------------------------------------------
    # Method/technique comparisons
    # --------------------------------------------------------
    # Aggregate complementary evidence within each document.
    # This permits, for example, one DP passage to describe clipping/noise and
    # another to explain the privacy purpose, while still requiring both facets
    # to be present somewhere in that paper's retrieved evidence.
    if method_question:
        per_document: dict[str, dict[str, object]] = {}

        for item in results:
            if float(item.get("noise_penalty", 0)) >= 3.0:
                continue

            text = unicodedata.normalize("NFKC", item.get("text", ""))
            tokens = set(tokenize(text, remove_stopwords=False))

            if not anchors.issubset(tokens):
                continue

            methods, purposes = method_purpose_score(text)
            if not methods and not purposes:
                continue

            document = str(item.get("document", "Unknown paper"))
            state = per_document.setdefault(
                document,
                {"methods": 0, "purposes": 0, "topic_match": False, "relevant": False},
            )

            state["methods"] = int(state["methods"]) + methods
            state["purposes"] = int(state["purposes"]) + purposes

            topic_overlap = bool(topics & tokens) if topics else True
            state["topic_match"] = bool(state["topic_match"]) or topic_overlap

            semantic_support = (
                item.get("retrieval_method") in {"semantic", "hybrid"}
                and float(item.get("semantic_score", 0)) >= 0.25
            )
            lexical_support = (
                float(item.get("coverage", 0)) >= 0.14
                and float(item.get("bm25_score", item.get("score", 0))) > 0
            )
            representative_support = bool(item.get("comparison_representative"))
            hybrid_support = float(item.get("hybrid_score", 0)) > 0

            state["relevant"] = bool(state["relevant"]) or any((
                semantic_support, lexical_support, representative_support, hybrid_support
            ))

        qualified_documents = {
            document
            for document, state in per_document.items()
            if int(state["methods"]) > 0
            and int(state["purposes"]) > 0
            and bool(state["relevant"])
            and (not topics or bool(state["topic_match"]))
        }

        return len(qualified_documents) >= 2

    # --------------------------------------------------------
    # General cross-paper comparisons
    # --------------------------------------------------------
    documents = set()

    for item in results:
        if float(item.get("noise_penalty", 0)) >= 3.0:
            continue

        tokens = set(tokenize(
            unicodedata.normalize("NFKC", item.get("text", "")),
            remove_stopwords=False,
        ))

        if not anchors.issubset(tokens):
            continue

        topic_coverage = len(topics & tokens) / len(topics) if topics else 1.0
        semantic_support = (
            item.get("retrieval_method") in {"semantic", "hybrid"}
            and float(item.get("semantic_score", 0)) >= 0.30
        )
        representative_support = (
            not topics
            and bool(item.get("comparison_representative"))
            and item.get("retrieval_method") in {"semantic", "hybrid"}
            and (float(item.get("hybrid_score", 0)) > 0
                 or float(item.get("semantic_score", 0)) >= 0.05)
        )

        if (representative_support
                or (bool(topics) and topic_coverage >= 0.50)
                or (semantic_support and bool(topics & tokens))):
            documents.add(item["document"])

    return len(documents) >= 2


def methodology_answer_is_complete(answer: str, results: list[dict]) -> bool:
    """Guard the supported private-SGD outline against loss of its key operations.

    Only activate for evidence explicitly describing that pipeline. This is a
    targeted regression guard, not a general factual-entailment evaluator.
    """
    outlines = [r['text'].lower() for r in results
                if methodology_score(r.get('text', '')) >= 2]
    if not any('clip' in text and 'add noise' in text and 'privacy accountant' in text
               for text in outlines):
        return True
    low = answer.lower()
    return all(any(term in low for term in alternatives) for alternatives in (
        ('gradient',), ('clip', 'bound'), ('average', 'averaging', 'aggregate'),
        ('noise',), ('update', 'step'), ('accountant', 'privacy loss', 'privacy cost'),
    ))
