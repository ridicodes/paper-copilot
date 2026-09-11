from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np
from pathlib import Path
from typing import Any
import json
import math
import pickle
import re

from rank_bm25 import BM25Okapi


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_VERSION = 6


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

    embeddings = build_embeddings([chunk["text"] for chunk in all_chunks])

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

    np.save(idx_dir / "embeddings.npy", embeddings, allow_pickle=False)

    # --------------------------------------------------------
    # Store metadata separately
    # --------------------------------------------------------

    meta = {
        "version": INDEX_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
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

    if not query or k <= 0:
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
        result["chunk_index"] = chunk_index
        result["bm25_score"] = bm25_score
        result["retrieval_method"] = "bm25"

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


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load text embeddings only when requested; prefer the local cache."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except OSError:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_embeddings(texts: list[str]) -> np.ndarray:
    return np.asarray(get_embedding_model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ), dtype=np.float32)


def load_embeddings(idx_dir: str | Path) -> np.ndarray:
    directory = Path(idx_dir)
    with (directory / "meta.json").open(encoding="utf-8") as file:
        meta = json.load(file)
    if not isinstance(meta, dict) or meta.get("embedding_model") != EMBEDDING_MODEL_NAME:
        raise ValueError("Semantic index is missing or incompatible. Re-index the library.")
    embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
    if (embeddings.ndim != 2
            or embeddings.shape != (len(meta["chunks"]), meta.get("embedding_dimension"))
            or not np.isfinite(embeddings).all()):
        raise ValueError("Invalid embedding matrix. Re-index the library.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Empty embedding vectors. Re-index the library.")
    return embeddings / norms


def _semantic_candidates(idx_dir, query, chunks):
    embeddings = load_embeddings(idx_dir)
    query_embedding = build_embeddings([query])[0]
    if embeddings.shape != (len(chunks), len(query_embedding)):
        raise ValueError("Embeddings do not match the library. Re-index the library.")
    scores = embeddings @ query_embedding
    candidates = []
    for index, chunk in enumerate(chunks):
        coverage, matched = calculate_coverage(query, chunk["text"])
        noise = calculate_noise_penalty(chunk["text"])
        # A repeated journal header is milder noise than a bibliography.
        header_hits = sum(term in chunk["text"].lower() for term in ("issn", "international journal"))
        if header_hits:
            noise = noise - 1.5 * header_hits + 0.5
        # Bibliographies often have many years, even without a References heading.
        years = len(re.findall(r"\b(?:19|20)\d{2}\b", chunk["text"]))
        noise += 3.0 if years >= 4 else 0.0
        score = float(scores[index])
        candidates.append(dict(
            chunk, chunk_index=index, semantic_score=score, score=score,
            rerank_score=score - 0.08 * noise, coverage=coverage,
            matched_terms=matched, noise_penalty=noise,
            retrieval_method="semantic",
        ))
    return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)


def _diverse_results(candidates, k, duplicate_threshold, max_per_page):
    selected = []
    pages = Counter()
    for candidate in candidates:
        page_key = (candidate["document"], candidate["page"])
        if pages[page_key] >= max_per_page:
            continue
        if any(token_similarity(candidate["text"], other["text"]) >= duplicate_threshold
               for other in selected):
            continue
        selected.append(dict(candidate, rank=len(selected) + 1))
        pages[page_key] += 1
        if len(selected) >= k:
            break
    return selected


def semantic_search(idx_dir, query, k=5, min_score=0.0,
                    candidate_multiplier=8, duplicate_threshold=0.82, max_per_page=2):
    if not query.strip() or k <= 0:
        return []
    _, chunks = load_index(idx_dir)
    candidates = _semantic_candidates(idx_dir, query, chunks)
    return _diverse_results(
        [item for item in candidates if item["semantic_score"] >= min_score],
        k, duplicate_threshold, max_per_page,
    )


def hybrid_search(idx_dir, query, k=5, min_score=0.0,
                  candidate_multiplier=8, duplicate_threshold=0.82, max_per_page=2):
    """Fuse quality-adjusted ranks, applying diversity only after fusion."""
    if not query.strip() or k <= 0:
        return []
    _, chunks = load_index(idx_dir)
    depth = min(len(chunks), max(40, k * candidate_multiplier))
    # Keep unfiltered lexical candidates: no page cap or duplicate removal yet.
    lexical = search(idx_dir, query, k=len(chunks), min_score=min_score,
                     duplicate_threshold=2.0, max_per_page=len(chunks))
    lexical = [item for item in lexical if item["bm25_score"] > 0][:depth]
    semantic = _semantic_candidates(idx_dir, query, chunks)
    semantic_by_index = {item["chunk_index"]: item for item in semantic}
    semantic = [item for item in semantic if item["semantic_score"] >= min_score][:depth]
    merged = {}
    for method, ranking in (("bm25", lexical), ("semantic", semantic)):
        for rank, item in enumerate(ranking, 1):
            index = item["chunk_index"]
            result = merged.setdefault(index, dict(
                semantic_by_index[index], bm25_score=0.0, bm25_rank=None,
                semantic_rank=None, rrf_score=0.0, retrieval_method="hybrid",
            ))
            result[method + "_rank"] = rank
            result["rrf_score"] += 1.0 / (60 + rank)
            if method == "bm25":
                result["bm25_score"] = item["bm25_score"]
                result["intent_bonus"] = item["intent_bonus"]
    for result in merged.values():
        result["hybrid_score"] = result["rrf_score"] / (1 + 0.15 * result["noise_penalty"])
        if is_methodology_question(query):
            result["hybrid_score"] *= 1 + 0.25 * methodology_score(result["text"])
        result["score"] = result["hybrid_score"]
        result["rerank_score"] = result["hybrid_score"]
    candidates = sorted(merged.values(), key=lambda item: item["hybrid_score"], reverse=True)
    if is_comparison_question(query):
        # A comparison cannot be answered from a top-k list monopolized by one
        # paper. Lead with each document's strongest candidate, then preserve
        # the fused order for all remaining passages.
        first_by_document = {}
        for candidate in candidates:
            first_by_document.setdefault(candidate["document"], candidate)
        leaders = list(first_by_document.values())
        leader_ids = {item["chunk_index"] for item in leaders}
        candidates = leaders + [item for item in candidates
                                if item["chunk_index"] not in leader_ids]
    return _diverse_results(candidates, k, duplicate_threshold, max_per_page)


def is_comparison_question(query: str) -> bool:
    low = query.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in (
        "compare", "comparison", "differently", "difference", "differences",
        "both papers", "two papers", "between the papers", "across papers",
    ))


def is_methodology_question(query: str) -> bool:
    if is_comparison_question(query):
        return False
    low = query.lower()
    if re.search(r"\b(method|methodology|algorithm|procedure|operation|perform|accumulat)\w*\b", low):
        return True
    if low.startswith("how can"):
        return bool(re.search(r"\b(train|learn|privacy|private|protect|gradient)\w*\b", low))
    return bool(re.search(r"\bhow (?:do|does|are|is)\b", low))


def methodology_score(text: str) -> int:
    """Prefer concrete procedure descriptions over feasibility or related work."""
    return sum(bool(re.search(pattern, text, re.I)) for pattern in (
        r"outlines our basic method", r"at each step", r"algorithm \d",
        r"we compute", r"we .*?update", r"next we describe",
    ))
