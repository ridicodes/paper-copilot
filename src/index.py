from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import json
import math
import pickle
import re

from rank_bm25 import BM25Okapi


# ============================================================
# WEEK 5 CONFIGURATION
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
}


GENERIC_QUERY_WORDS = {
    "paper",
    "study",
    "article",
    "research",
    "review",
    "authors",
    "author",
    "according",
    "described",
    "discussed",
    "explain",
    "explains",
    "main",
}


BAD_PATTERNS = [
    "issn",
    "international journal",
    "copyright",
    "all rights reserved",
    "references",
    "bibliography",
]


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(
    text: str,
    remove_stopwords: bool = True,
) -> list[str]:
    """
    Convert text into normalized BM25 tokens.
    """

    tokens = re.findall(
        r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*",
        text.lower(),
    )

    if remove_stopwords:
        tokens = [
            token
            for token in tokens
            if token not in STOPWORDS
        ]

    return tokens


def query_terms(
    query: str,
) -> list[str]:
    """
    Return meaningful terms used for coverage and reranking.
    """

    terms = tokenize(
        query,
        remove_stopwords=True,
    )

    return [
        term
        for term in terms
        if term not in GENERIC_QUERY_WORDS
        and (len(term) > 2 or term.isdigit())
    ]


# ============================================================
# JSON LOADING
# ============================================================

def _extract_chunks_from_json(
    data: Any,
) -> list[dict]:
    """
    Accept a few common ingestion JSON layouts.

    Supported examples:

    [
        {"page": 1, "text": "..."},
        ...
    ]

    or:

    {
        "chunks": [
            {"page": 1, "text": "..."},
            ...
        ]
    }

    or:

    {
        "pages": [...]
    }
    """

    if isinstance(data, list):
        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    if isinstance(data, dict):

        for key in [
            "chunks",
            "documents",
            "pages",
            "items",
        ]:

            value = data.get(key)

            if isinstance(value, list):
                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

    raise ValueError(
        "Could not find a list of chunks in the ingestion JSON."
    )


def load_chunks(
    json_path: str | Path,
) -> list[dict]:
    """
    Load chunks from one paper and attach document identity.

    Week 5 addition:
    every chunk receives a `document` field.
    """

    json_path = Path(
        json_path
    )

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    raw_chunks = (
        _extract_chunks_from_json(
            data
        )
    )

    document_name = (
        json_path.stem
        + ".pdf"
    )

    normalized_chunks: list[dict] = []

    for position, chunk in enumerate(
        raw_chunks
    ):

        text = str(
            chunk.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        try:
            page = int(
                chunk.get(
                    "page",
                    1,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            page = 1

        normalized = dict(
            chunk
        )

        normalized[
            "text"
        ] = text

        normalized[
            "page"
        ] = page

        # Preserve an existing source/document field
        # if ingest.py already supplied one.
        existing_document = (
            chunk.get(
                "document"
            )
            or chunk.get(
                "source"
            )
            or chunk.get(
                "filename"
            )
        )

        normalized[
            "document"
        ] = (
            str(existing_document)
            if existing_document
            else document_name
        )

        normalized[
            "document_stem"
        ] = Path(
            normalized[
                "document"
            ]
        ).stem

        normalized.setdefault(
            "chunk_id",
            f"{json_path.stem}_{position}",
        )

        normalized_chunks.append(
            normalized
        )

    return normalized_chunks


# ============================================================
# INDEX BUILDING
# ============================================================

def build_index(
    json_paths: str
    | Path
    | list[str]
    | list[Path],
    idx_dir: str | Path,
) -> None:
    """
    Build one BM25 index across one or more papers.

    Backwards compatible:

        build_index("paper.json", "outputs/paper")

    Week 5:

        build_index(
            [
                "outputs/paper_a.json",
                "outputs/paper_b.json",
            ],
            "outputs/library_index",
        )
    """

    if isinstance(
        json_paths,
        (str, Path),
    ):

        json_paths = [
            json_paths
        ]

    paths = [
        Path(path)
        for path in json_paths
    ]

    if not paths:
        raise ValueError(
            "No JSON files were supplied to build_index()."
        )

    all_chunks: list[dict] = []

    for path in paths:

        if not path.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {path}"
            )

        chunks = load_chunks(
            path
        )

        all_chunks.extend(
            chunks
        )

    if not all_chunks:
        raise ValueError(
            "No usable text chunks were found."
        )

    corpus_tokens = [
        tokenize(
            chunk["text"],
            remove_stopwords=True,
        )
        for chunk in all_chunks
    ]

    bm25 = BM25Okapi(
        corpus_tokens
    )

    idx_dir = Path(
        idx_dir
    )

    idx_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Store BM25 object
    # --------------------------------------------------------

    with (
        idx_dir
        / "bm25.pkl"
    ).open(
        "wb"
    ) as file:

        pickle.dump(
            bm25,
            file,
        )

    # --------------------------------------------------------
    # Store metadata separately
    # --------------------------------------------------------

    meta = {
        "version": 5,
        "num_documents": len(
            {
                chunk["document"]
                for chunk in all_chunks
            }
        ),
        "num_chunks": len(
            all_chunks
        ),
        "documents": sorted(
            {
                chunk["document"]
                for chunk in all_chunks
            }
        ),
        "chunks": all_chunks,
    }

    with (
        idx_dir
        / "meta.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            meta,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# INDEX LOADING
# ============================================================

def load_index(
    idx_dir: str | Path,
) -> tuple[BM25Okapi, list[dict]]:
    """
    Load BM25 and chunk metadata.
    """

    idx_dir = Path(
        idx_dir
    )

    bm25_path = (
        idx_dir
        / "bm25.pkl"
    )

    meta_path = (
        idx_dir
        / "meta.json"
    )

    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {bm25_path}"
        )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Index metadata not found: {meta_path}"
        )

    with bm25_path.open(
        "rb"
    ) as file:

        bm25 = pickle.load(
            file
        )

    with meta_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        meta = json.load(
            file
        )

    # Week 5 format.
    if isinstance(
        meta,
        dict,
    ) and isinstance(
        meta.get(
            "chunks"
        ),
        list,
    ):

        chunks = meta[
            "chunks"
        ]

    # Backward compatibility with an older metadata file
    # that may itself have been a list.
    elif isinstance(
        meta,
        list,
    ):

        chunks = meta

    else:

        raise ValueError(
            "Invalid meta.json format."
        )

    return (
        bm25,
        chunks,
    )


# ============================================================
# RERANKING HELPERS
# ============================================================

def calculate_coverage(
    query: str,
    text: str,
) -> tuple[
    float,
    list[str],
]:
    """
    Measure how many meaningful query terms occur
    in the evidence passage.
    """

    terms = query_terms(
        query
    )

    if not terms:
        return (
            0.0,
            [],
        )

    text_tokens = set(
        tokenize(
            text,
            remove_stopwords=False,
        )
    )

    matched_terms = [
        term
        for term in terms
        if term in text_tokens
    ]

    coverage = (
        len(
            set(
                matched_terms
            )
        )
        / len(
            set(
                terms
            )
        )
    )

    return (
        coverage,
        sorted(
            set(
                matched_terms
            )
        ),
    )


def calculate_intent_bonus(
    query: str,
    text: str,
) -> float:
    """
    Give extra weight when a passage explicitly matches
    the type of answer being requested.
    """

    q_low = query.lower()
    text_low = text.lower()

    bonus = 0.0

    intent_groups = {
        "stages": [
            "stage",
            "stages",
        ],
        "steps": [
            "step",
            "steps",
        ],
        "weakness": [
            "weakness",
            "weaknesses",
        ],
        "limitation": [
            "limitation",
            "limitations",
            "limited",
        ],
        "advantage": [
            "advantage",
            "advantages",
        ],
        "application": [
            "application",
            "applications",
        ],
        "challenge": [
            "challenge",
            "challenges",
        ],
    }

    for query_word, text_words in (
        intent_groups.items()
    ):

        if query_word in q_low:

            if any(
                word in text_low
                for word in text_words
            ):
                bonus += 3.0

    # Handle plural/singular variations in the query itself.
    if (
        "weaknesses" in q_low
        and "weakness" in text_low
    ):
        bonus += 3.0

    if (
        "limitations" in q_low
        and "limitation" in text_low
    ):
        bonus += 3.0

    if (
        "applications" in q_low
        and "application" in text_low
    ):
        bonus += 3.0

    return bonus


def calculate_noise_penalty(
    text: str,
) -> float:

    low = text.lower()

    penalty = 0.0

    for pattern in BAD_PATTERNS:

        if pattern in low:
            penalty += 1.5

    # Reference-heavy / bibliography-style text.
    citation_count = len(
        re.findall(
            r"\[\d+\]",
            text,
        )
    )

    if citation_count >= 4:
        penalty += 1.5

    url_count = len(
        re.findall(
            r"https?://|www\.",
            low,
        )
    )

    if url_count >= 1:
        penalty += 1.0

    return penalty


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def token_similarity(
    text_a: str,
    text_b: str,
) -> float:

    tokens_a = set(
        tokenize(
            text_a
        )
    )

    tokens_b = set(
        tokenize(
            text_b
        )
    )

    if (
        not tokens_a
        or not tokens_b
    ):
        return 0.0

    intersection = len(
        tokens_a
        & tokens_b
    )

    union = len(
        tokens_a
        | tokens_b
    )

    if union == 0:
        return 0.0

    return (
        intersection
        / union
    )


# ============================================================
# SEARCH
# ============================================================

def search(
    idx_dir: str | Path,
    query: str,
    k: int = 5,
    min_score: float = 0.0,
    candidate_multiplier: int = 8,
    duplicate_threshold: float = 0.82,
    max_per_page: int = 2,
) -> list[dict]:
    """
    Search one combined research-library index.

    Week 5 result fields include:

        result["document"]
        result["page"]
        result["text"]

    along with the Week 3 retrieval diagnostics.
    """

    query = query.strip()

    if not query:
        return []

    bm25, chunks = load_index(
        idx_dir
    )

    if not chunks:
        return []

    tokens = tokenize(
        query,
        remove_stopwords=True,
    )

    if not tokens:
        tokens = tokenize(
            query,
            remove_stopwords=False,
        )

    raw_scores = bm25.get_scores(
        tokens
    )

    candidate_count = min(
        len(chunks),
        max(
            int(k)
            * int(
                candidate_multiplier
            ),
            int(k),
        ),
    )

    candidate_indices = sorted(
        range(
            len(
                raw_scores
            )
        ),
        key=lambda index: raw_scores[
            index
        ],
        reverse=True,
    )[
        :candidate_count
    ]

    candidates: list[dict] = []

    for chunk_index in (
        candidate_indices
    ):

        bm25_score = float(
            raw_scores[
                chunk_index
            ]
        )

        if bm25_score < min_score:
            continue

        chunk = chunks[
            chunk_index
        ]

        text = str(
            chunk.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        coverage, matched_terms = (
            calculate_coverage(
                query,
                text,
            )
        )

        intent_bonus = (
            calculate_intent_bonus(
                query,
                text,
            )
        )

        noise_penalty = (
            calculate_noise_penalty(
                text
            )
        )

        # ----------------------------------------------------
        # Week 3 reranking
        #
        # BM25 remains the foundation.
        # Query coverage adds a strong relevance signal.
        # Intent helps list/process/limitation questions.
        # Noise reduces bibliography/header passages.
        # ----------------------------------------------------

        rerank_score = (
            bm25_score
            + (
                coverage
                * 3.0
            )
            + intent_bonus
            - noise_penalty
        )

        result = dict(
            chunk
        )

        result[
            "score"
        ] = bm25_score

        result[
            "rerank_score"
        ] = rerank_score

        result[
            "coverage"
        ] = coverage

        result[
            "matched_terms"
        ] = matched_terms

        result[
            "intent_bonus"
        ] = intent_bonus

        result[
            "noise_penalty"
        ] = noise_penalty

        # Ensure every Week 5 result identifies its paper.
        result.setdefault(
            "document",
            "Unknown paper",
        )

        candidates.append(
            result
        )

    # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            item[
                "rerank_score"
            ],
            item[
                "score"
            ],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Duplicate suppression + diversity
    #
    # Important Week 5 change:
    # per-page limits are tracked PER DOCUMENT.
    #
    # Page 2 of Paper A is not the same as
    # Page 2 of Paper B.
    # --------------------------------------------------------

    selected: list[dict] = []

    page_counts: Counter[
        tuple[str, int]
    ] = Counter()

    for candidate in candidates:

        document = str(
            candidate.get(
                "document",
                "Unknown paper",
            )
        )

        page = int(
            candidate.get(
                "page",
                1,
            )
        )

        page_key = (
            document,
            page,
        )

        if (
            page_counts[
                page_key
            ]
            >= max_per_page
        ):
            continue

        duplicate = False

        for existing in selected:

            # Duplicate suppression should compare the text.
            # It is allowed to suppress highly overlapping
            # chunks even across papers.
            similarity = (
                token_similarity(
                    candidate[
                        "text"
                    ],
                    existing[
                        "text"
                    ],
                )
            )

            if (
                similarity
                >= duplicate_threshold
            ):

                duplicate = True
                break

        if duplicate:
            continue

        selected.append(
            candidate
        )

        page_counts[
            page_key
        ] += 1

        if len(
            selected
        ) >= int(k):
            break

    # --------------------------------------------------------
    # Final rank numbers
    # --------------------------------------------------------

    for rank, result in enumerate(
        selected,
        start=1,
    ):

        result[
            "rank"
        ] = rank

    return selected