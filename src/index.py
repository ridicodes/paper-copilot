import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "he", "her", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "may", "might", "more", "most", "of",
    "on", "or", "our", "ours", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these", "they",
    "this", "those", "to", "too", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "you", "your",
    "yours",
}

QUESTION_NOISE = {
    "paper", "study", "article", "described", "discussed", "explain", "explains",
    "main", "according", "authors", "review", "research"
}

TABLE_NOISE_PATTERNS = [
    "no researchers usefulness definition",
    "table 1",
    "table 2",
    "references",
    "issn",
    "vol.",
    "doi:",
]


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [
        token
        for token in tokens
        if token not in STOPWORDS and (len(token) > 2 or token.isdigit())
    ]


def _query_tokens(text: str) -> list[str]:
    """
    Tokenize a question, removing generic research-question wording that should
    not dominate retrieval.
    """
    return [
        token
        for token in _tokenize(text)
        if token not in QUESTION_NOISE
    ]


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _contains_any(text: str, terms: set[str]) -> int:
    low = text.lower()
    return sum(1 for term in terms if term in low)


def _intent_bonus(query: str, text: str) -> float:
    """
    Small rule-based bonuses for common research-paper question intents.

    This stays lightweight and lexical; semantic embeddings are intentionally
    deferred to a later project week.
    """
    q = query.lower()
    t = text.lower()
    bonus = 0.0

    # List / process / sequence questions
    if any(word in q for word in ["stages", "steps", "phases", "process"]):
        if any(phrase in t for phrase in [
            "stages of", "steps of", "steps are", "stages are",
            "in more detail", "1)", "2)", "3)"
        ]):
            bonus += 3.0

    # Limitation / weakness questions
    if any(word in q for word in ["weakness", "weaknesses", "limitations", "drawbacks", "problems"]):
        if any(word in t for word in [
            "weakness", "weaknesses", "limitation", "limitations",
            "drawback", "problem"
        ]):
            bonus += 2.5

    # Method questions
    if any(word in q for word in ["method", "methods", "approach", "approaches", "technique", "techniques"]):
        if any(word in t for word in [
            "method", "methods", "approach", "approaches",
            "technique", "techniques"
        ]):
            bonus += 1.5

    # Dataset questions
    if any(word in q for word in ["dataset", "datasets", "data"]):
        if any(word in t for word in ["dataset", "datasets", "data"]):
            bonus += 1.5

    return bonus


def _noise_penalty(text: str) -> float:
    """
    Penalize obvious table/reference/header chunks that often get accidental
    lexical matches but make poor evidence.
    """
    low = " ".join(text.lower().split())
    penalty = 0.0

    for pattern in TABLE_NOISE_PATTERNS:
        if pattern in low:
            penalty += 1.0

    # Reference-like chunks often contain many bracketed citation numbers.
    citation_count = len(re.findall(r"\[\d+\]", text))
    if citation_count >= 5:
        penalty += 1.5

    return min(penalty, 3.0)


def build_index(json_path: str | Path, out_dir: str | Path) -> Path:
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])

    tokenized_corpus = [_tokenize(c.get("text", "")) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    (out_dir / "meta.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir


def search(
    out_dir: str | Path,
    query: str,
    k: int = 5,
    min_score: float = 0.0,
    candidate_multiplier: int = 8,
    duplicate_threshold: float = 0.82,
    max_per_page: int = 2,
) -> list[dict]:
    out_dir = Path(out_dir)

    with open(out_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    chunks = meta.get("chunks", [])

    q_tokens = _query_tokens(query)
    if not q_tokens or not chunks:
        return []

    scores = bm25.get_scores(q_tokens)

    pool_size = min(
        len(chunks),
        max(int(k), int(k) * max(1, int(candidate_multiplier))),
    )
    candidate_idx = np.argsort(scores)[::-1][:pool_size]

    q_unique = set(q_tokens)
    normalized_query = " ".join(query.lower().split())

    candidates: list[dict] = []

    for idx in candidate_idx:
        idx = int(idx)
        bm25_score = float(scores[idx])

        if bm25_score <= min_score:
            continue

        chunk = chunks[idx]
        text = chunk.get("text", "")
        chunk_tokens = _token_set(text)

        matched_terms = q_unique & chunk_tokens
        coverage = len(matched_terms) / len(q_unique) if q_unique else 0.0

        normalized_text = " ".join(text.lower().split())
        phrase_bonus = (
            1.0
            if len(normalized_query) >= 5 and normalized_query in normalized_text
            else 0.0
        )

        intent_bonus = _intent_bonus(query, text)
        noise_penalty = _noise_penalty(text)

        rerank_score = (
            bm25_score
            + (2.0 * coverage)
            + (0.75 * phrase_bonus)
            + intent_bonus
            - noise_penalty
        )

        candidates.append(
            {
                "score": bm25_score,
                "rerank_score": rerank_score,
                "coverage": coverage,
                "matched_terms": sorted(matched_terms),
                "intent_bonus": intent_bonus,
                "noise_penalty": noise_penalty,
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "text": text,
                "_token_set": chunk_tokens,
            }
        )

    candidates.sort(
        key=lambda item: (item["rerank_score"], item["score"]),
        reverse=True,
    )

    results: list[dict] = []
    selected_token_sets: list[set[str]] = []
    page_counts: dict[int, int] = {}

    for candidate in candidates:
        if len(results) >= int(k):
            break

        page = int(candidate["page"])

        if page_counts.get(page, 0) >= max_per_page:
            continue

        is_near_duplicate = any(
            _jaccard_similarity(candidate["_token_set"], selected_set)
            >= duplicate_threshold
            for selected_set in selected_token_sets
        )
        if is_near_duplicate:
            continue

        selected_token_sets.append(candidate["_token_set"])
        page_counts[page] = page_counts.get(page, 0) + 1

        result = {
            key: value
            for key, value in candidate.items()
            if key != "_token_set"
        }
        result["rank"] = len(results) + 1
        results.append(result)

    return results
