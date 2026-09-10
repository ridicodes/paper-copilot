from pathlib import Path
import html
import re

import fitz
import streamlit as st

from src.ingest import ingest_pdf
from src.index import build_index, search, semantic_search, hybrid_search
from src.evidence import (
    evidence_is_sufficient as retrieval_is_sufficient,
    passage_is_relevant, supported_passages, methodology_answer_is_complete,
)
from src.llm import ollama_chat
from src.citations import citations_are_complete, normalize_answer
from src.index import is_methodology_question, methodology_score


# ============================================================
# CONFIGURATION
# ============================================================

BAD_PATTERNS = [
    "issn",
    "international journal",
    "copyright",
    "all rights reserved",
    "no researchers usefulness definition",
]

MIN_ANSWER_COVERAGE = 0.50
MAX_ANSWER_PASSAGES = 4

# Comparison questions search more deeply than the visible
# number of evidence cards.
COMPARISON_SEARCH_K = 20

LIBRARY_INDEX_DIR = Path("outputs") / "library_index"


# ============================================================
# QUERY HELPERS
# ============================================================

def query_keywords(query: str) -> list[str]:
    generic = {
        "paper",
        "papers",
        "study",
        "studies",
        "article",
        "articles",
        "described",
        "discussed",
        "explain",
        "explains",
        "main",
        "according",
        "authors",
        "author",
        "review",
        "research",
        "what",
        "which",
        "where",
        "when",
        "does",
        "this",
        "that",
        "with",
        "from",
        "into",
        "about",
        "both",
        "two",
    }

    return [
        word.lower()
        for word in re.findall(
            r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*",
            query,
        )
        if (len(word) > 2 or word.isdigit())
        and word.lower() not in generic
    ]


def is_cross_document_question(
    query: str,
) -> bool:
    q_low = query.lower()

    comparison_phrases = [
        "compare",
        "comparison",
        "compared",
        "difference",
        "differences",
        "different",
        "differently",
        "similar",
        "similarity",
        "similarities",
        "both papers",
        "two papers",
        "across the papers",
        "across papers",
        "between the papers",
        "between these papers",
        "how do the papers",
        "how do these papers",
    ]

    return any(
        phrase in q_low
        for phrase in comparison_phrases
    )


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not cleaned:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+|(?=\b\d+\)\s*)",
        cleaned,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def pick_snippet(
    text: str,
    keywords: list[str],
    query: str = "",
    max_len: int = 420,
) -> str:
    sentences = split_sentences(text)

    if not sentences:
        return ""

    q_low = query.lower()

    def is_bad(sentence: str) -> bool:
        low = sentence.lower()

        return any(
            pattern in low
            for pattern in BAD_PATTERNS
        )

    best_index = 0
    best_score = float("-inf")

    for i, sentence in enumerate(sentences):
        if is_bad(sentence):
            continue

        low = sentence.lower()
        score = 0.0

        for keyword in keywords:
            if keyword in low:
                score += 1.0

        if "stages" in q_low and "stages" in low:
            score += 4.0

        if "steps" in q_low and "steps" in low:
            score += 4.0

        if (
            any(
                word in q_low
                for word in [
                    "weakness",
                    "weaknesses",
                ]
            )
            and "weakness" in low
        ):
            score += 4.0

        if (
            "limitations" in q_low
            and "limitation" in low
        ):
            score += 4.0

        # For comparison questions, prefer sentences
        # that describe what the paper actually does.
        if is_cross_document_question(query):
            representative_phrases = [
                "our approach",
                "our work",
                "we develop",
                "we propose",
                "we use",
                "we train",
                "we apply",
                "we evaluate",
                "main components",
                "main component",
                "purpose",
                "goal",
                "machine learning",
                "training",
                "prediction",
                "recognition",
                "classification",
                "privacy",
                "differentially private",
                "image processing",
                "computer vision",
            ]

            for phrase in representative_phrases:
                if phrase in low:
                    score += 1.0

        if score > best_score:
            best_score = score
            best_index = i

    chosen = [
        sentences[best_index]
    ]

    list_intents = [
        "stages",
        "steps",
        "phases",
        "process",
        "weakness",
        "weaknesses",
        "limitation",
        "limitations",
        "advantage",
        "advantages",
        "disadvantage",
        "disadvantages",
        "challenge",
        "challenges",
        "application",
        "applications",
    ]

    if any(
        word in q_low
        for word in list_intents
    ):
        j = best_index + 1

        while (
            j < len(sentences)
            and len(" ".join(chosen)) < max_len
        ):
            if (
                re.match(
                    r"^\d+\)",
                    sentences[j],
                )
                or len(chosen) < 3
            ):
                chosen.append(
                    sentences[j]
                )
                j += 1
            else:
                break

    snippet = " ".join(
        chosen
    ).strip()

    if len(snippet) > max_len:
        snippet = (
            snippet[:max_len].rstrip()
            + "…"
        )

    return snippet


# ============================================================
# EVIDENCE HELPERS
# ============================================================

def evidence_strength(
    coverage: float,
) -> str:
    if coverage >= 0.75:
        return "Strong"

    if coverage >= 0.50:
        return "Moderate"

    return "Weak"


def highlight_terms(
    text: str,
    terms: list[str],
) -> str:
    escaped = html.escape(text)

    unique_terms = sorted(
        {
            term
            for term in terms
            if term.strip()
        },
        key=len,
        reverse=True,
    )

    for term in unique_terms:
        pattern = re.compile(
            rf"\b({re.escape(html.escape(term))})\b",
            re.IGNORECASE,
        )

        escaped = pattern.sub(
            r"<mark>\1</mark>",
            escaped,
        )

    return escaped


def overall_evidence_coverage(
    results: list[dict],
) -> float:
    if not results:
        return 0.0

    return max(
        float(
            result.get(
                "coverage",
                0.0,
            )
        )
        for result in results
    )


def evidence_is_sufficient(results: list[dict], question: str = "") -> bool:
    return retrieval_is_sufficient(
        question, results, comparison=is_cross_document_question(question),
    )


def get_matched_query_terms(
    result: dict,
    keywords: list[str],
) -> set[str]:
    matched = {
        str(term).lower()
        for term in result.get(
            "matched_terms",
            [],
        )
    }

    text = str(
        result.get(
            "text",
            "",
        )
    ).lower()

    for keyword in keywords:
        if keyword in text:
            matched.add(keyword)

    return matched


# ============================================================
# REPRESENTATIVE PASSAGE SCORING
# ============================================================

def representative_passage_score(
    result: dict,
    query: str,
) -> float:
    """
    Score how useful a passage is for explaining
    what a paper actually does.

    Used mainly for comparison questions.
    """

    text = str(
        result.get(
            "text",
            "",
        )
    ).lower()

    coverage = float(
        result.get(
            "coverage",
            0.0,
        )
    )

    rerank_score = float(
        result.get(
            "rerank_score",
            0.0,
        )
    )

    score = (
        coverage * 4.0
        + rerank_score * 0.20
    )

    strong_phrases = {
        "in this paper": 6.0,
        "we combine": 6.0,
        "our approach": 4.0,
        "our work": 2.5,
        "we develop": 3.0,
        "we propose": 3.0,
        "we introduce": 3.0,
        "we present": 2.0,
        "we use": 1.5,
        "we apply": 1.5,
        "we train": 2.0,
        "we evaluate": 1.5,
        "main components": 5.0,
        "main component": 4.0,
        "consists of": 2.0,
        "purpose": 2.0,
        "goal": 2.0,
        "training": 1.5,
        "differentially private": 2.5,
        "machine learning": 1.5,
        "computer vision": 1.5,
        "image processing": 1.5,
        "recognition": 1.0,
        "prediction": 1.0,
        "classification": 1.0,
    }

    for phrase, bonus in (
        strong_phrases.items()
    ):
        if phrase in text:
            score += bonus

    # Penalize passages that look like related work,
    # references or isolated benchmark commentary.
    weak_phrases = {
        "related work": 2.5,
        "references": 3.0,
        "bibliography": 3.0,
        "serving as benchmarks": 1.5,
        "focus of active work": 1.0,
    }

    for phrase, penalty in (
        weak_phrases.items()
    ):
        if phrase in text:
            score -= penalty

    return score


# ============================================================
# CROSS-DOCUMENT EVIDENCE SELECTION
# ============================================================

def select_cross_document_evidence(
    query: str,
    results: list[dict],
    max_passages: int = MAX_ANSWER_PASSAGES,
) -> list[dict]:
    if not results:
        return []

    grouped: dict[
        str,
        list[dict],
    ] = {}

    document_order: list[str] = []

    for result in results:
        document = str(
            result.get(
                "document",
                "Unknown paper",
            )
        )

        if document not in grouped:
            grouped[
                document
            ] = []

            document_order.append(
                document
            )

        grouped[
            document
        ].append(
            result
        )

    # Rank passages INSIDE each paper according to how
    # representative they are of that paper's purpose/method.
    for document in grouped:
        grouped[
            document
        ].sort(
            key=lambda item: (
                representative_passage_score(
                    item,
                    query,
                ),
                float(
                    item.get(
                        "coverage",
                        0.0,
                    )
                ),
                float(
                    item.get(
                        "rerank_score",
                        0.0,
                    )
                ),
            ),
            reverse=True,
        )

    selected: list[dict] = []

    # First pass: guarantee the best representative
    # passage from every retrieved paper.
    for document in document_order:
        candidates = grouped[
            document
        ]

        if not candidates:
            continue

        selected.append(
            candidates[0]
        )

        if (
            len(selected)
            >= max_passages
        ):
            return selected

    # Second pass: one additional supporting passage
    # from each paper where useful.
    for document in document_order:
        candidates = grouped[
            document
        ]

        if len(candidates) < 2:
            continue

        for candidate in candidates[1:]:
            coverage = float(
                candidate.get(
                    "coverage",
                    0.0,
                )
            )

            representative_score = (
                representative_passage_score(
                    candidate,
                    query,
                )
            )

            if (
                coverage < 0.20
                and representative_score < 2.0
            ):
                continue

            selected.append(
                candidate
            )

            break

        if (
            len(selected)
            >= max_passages
        ):
            break

    return selected[
        :max_passages
    ]


# ============================================================
# NORMAL ANSWER-EVIDENCE SELECTION
# ============================================================

def select_answer_evidence(
    query: str,
    results: list[dict],
    max_passages: int = MAX_ANSWER_PASSAGES,
) -> list[dict]:
    if not results:
        return []

    results = supported_passages(query, results)
    if not results:
        return []

    if is_cross_document_question(
        query
    ):
        return (
            select_cross_document_evidence(
                query,
                results,
                max_passages=max_passages,
            )
        )

    definition = re.match(r"what is (.+?)(?: and |\?|$)", query.strip(), re.I)
    if definition:
        subject = re.escape(definition.group(1))
        direct = [r for r in results if re.search(subject + r"\s+(?:is|as|refers|means)\b", r["text"], re.I)]
        if direct:
            return direct[:max_passages]

    if is_methodology_question(query):
        procedures = [r for r in results if methodology_score(r["text"]) >= 2]
        if procedures:
            return sorted(procedures, key=lambda r: methodology_score(r["text"]), reverse=True)[:max_passages]
        return results[:max_passages]

    keywords = query_keywords(
        query
    )

    q_low = query.lower()

    multi_evidence_intents = [
        "weakness",
        "weaknesses",
        "limitation",
        "limitations",
        "advantage",
        "advantages",
        "disadvantage",
        "disadvantages",
        "application",
        "applications",
        "reason",
        "reasons",
        "challenge",
        "challenges",
        "problem",
        "problems",
    ]

    if any(
        intent in q_low
        for intent in multi_evidence_intents
    ):
        selected = []

        for result in results[
            :max_passages
        ]:
            if not selected:
                selected.append(
                    result
                )
                continue

            coverage = float(
                result.get(
                    "coverage",
                    0.0,
                )
            )

            if coverage >= 0.20:
                selected.append(
                    result
                )

        return selected

    required_terms = set(
        keywords
    )

    selected: list[dict] = []
    covered_terms: set[str] = set()

    for result in results:
        coverage = float(
            result.get(
                "coverage",
                0.0,
            )
        )

        if not passage_is_relevant(result):
            continue

        result_terms = (
            get_matched_query_terms(
                result,
                keywords,
            )
        )

        if not selected:
            selected.append(
                result
            )

            covered_terms.update(
                result_terms
            )

        else:
            new_terms = (
                result_terms
                - covered_terms
            )

            if new_terms:
                selected.append(
                    result
                )

                covered_terms.update(
                    result_terms
                )

        if (
            required_terms
            and required_terms.issubset(
                covered_terms
            )
        ):
            break

        if (
            len(selected)
            >= max_passages
        ):
            break

    if not selected:
        selected = [
            results[0]
        ]

    return selected


# ============================================================
# DOCUMENT HELPERS
# ============================================================

def document_display_name(
    document: str,
) -> str:
    return Path(
        document
    ).stem


def find_pdf_path(
    document: str,
) -> str | None:
    document_name = Path(
        document
    ).name

    for path in (
        st.session_state.get(
            "pdf_paths",
            [],
        )
        or []
    ):
        if (
            Path(path).name
            == document_name
        ):
            return str(path)

    return None


def citation_text(
    document: str,
    page: int,
) -> str:
    name = (
        document_display_name(
            document
        )
    )

    return (
        f"[{name}, p. {page}]"
    )


# ============================================================
# EVIDENCE-ID CITATION SYSTEM
# ============================================================

def build_evidence_map(
    results: list[dict],
) -> dict[str, dict]:
    evidence_map: dict[
        str,
        dict,
    ] = {}

    for index, result in enumerate(
        results,
        start=1,
    ):
        evidence_id = (
            f"E{index}"
        )

        evidence_map[
            evidence_id
        ] = result

    return evidence_map


def replace_evidence_ids_with_citations(
    answer: str,
    evidence_map: dict[str, dict],
) -> str:
    def replacement(
        match: re.Match,
    ) -> str:
        evidence_id = (
            match.group(1)
            .upper()
        )

        result = evidence_map.get(
            evidence_id
        )

        if result is None:
            return match.group(0)

        document = str(
            result.get(
                "document",
                "Unknown paper",
            )
        )

        page = int(
            result.get(
                "page",
                1,
            )
        )

        return citation_text(
            document,
            page,
        )

    return re.sub(
        r"\[(E\d+)\]",
        replacement,
        answer,
        flags=re.IGNORECASE,
    )


def extract_evidence_ids(
    answer: str,
) -> list[str]:
    return [
        evidence_id.upper()
        for evidence_id in re.findall(
            r"\[(E\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )
    ]


def evidence_ids_are_valid(
    answer: str,
    evidence_map: dict[str, dict],
) -> bool:
    used_ids = extract_evidence_ids(
        answer
    )

    if not used_ids:
        return False

    allowed_ids = set(
        evidence_map.keys()
    )

    return citations_are_complete(answer) and all(
        evidence_id in allowed_ids
        for evidence_id in used_ids
    )


# ============================================================
# EXTRACTIVE FALLBACK
# ============================================================

def make_extractive_answer(
    query: str,
    results: list[dict],
    max_points: int = 6,
) -> str:
    if not results:
        return (
            "Not found in the "
            "provided evidence."
        )

    keywords = query_keywords(
        query
    )

    points: list[str] = []

    seen: set[
        tuple[
            str,
            int,
            str,
        ]
    ] = set()

    for result in results:
        page = int(
            result.get(
                "page",
                1,
            )
        )

        document = str(
            result.get(
                "document",
                "Unknown paper",
            )
        )

        if is_methodology_question(query) and not is_cross_document_question(query):
            snippet = " ".join(re.split(r"(?<=[.!?])\s+", result.get("text", ""))[:3])
        else:
            snippet = pick_snippet(result.get("text", ""), keywords, query=query, max_len=1400)

        key = (
            document,
            page,
            snippet,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        citation = citation_text(
            document,
            page,
        )

        sentences = re.split(r"(?<=[.!?])\s+", snippet)
        points.append("- " + " ".join(f"{sentence} **{citation}**" for sentence in sentences))

        if (
            len(points)
            >= max_points
        ):
            break

    if not points:
        return (
            "Not found in the "
            "provided evidence."
        )

    return "\n".join(
        points
    )


# ============================================================
# PDF PAGE PREVIEW
# ============================================================

@st.cache_data
def render_page_png(
    pdf_path: str,
    page_num: int,
    zoom: float = 1.8,
) -> bytes:
    doc = fitz.open(
        pdf_path
    )

    try:
        page = doc.load_page(
            page_num - 1
        )

        matrix = fitz.Matrix(
            zoom,
            zoom,
        )

        pix = page.get_pixmap(
            matrix=matrix
        )

        return pix.tobytes(
            "png"
        )

    finally:
        doc.close()


def set_view_page(
    page: int,
    document: str,
) -> None:
    st.session_state[
        "view_page"
    ] = int(page)

    st.session_state[
        "view_document"
    ] = document


# ============================================================
# ANSWER PROMPT
# ============================================================

def build_answer_prompt(
    query: str,
    results: list[dict],
) -> tuple[
    str,
    dict[str, dict],
]:
    evidence_map = (
        build_evidence_map(
            results
        )
    )

    evidence_blocks: list[
        str
    ] = []

    for evidence_id, result in (
        evidence_map.items()
    ):
        page = int(
            result.get(
                "page",
                1,
            )
        )

        document = str(
            result.get(
                "document",
                "Unknown paper",
            )
        )

        display_name = (
            document_display_name(
                document
            )
        )

        # Keep complete page-bounded chunks: lexical snippets can omit the
        # procedural details that made semantic evidence useful.
        text = result.get("text", "").strip()

        evidence_blocks.append(
            (
                f"[{evidence_id}] "
                f"Source: {display_name}, "
                f"page {page}\n"
                f"{text}"
            )
        )

    evidence = "\n\n".join(
        evidence_blocks
    )

    comparison_instruction = ""

    if is_cross_document_question(
        query
    ):
        comparison_instruction = """
This is a cross-document comparison question.

COMPARISON RULES:

- Identify what EACH paper is primarily doing with the topic in the question.
- Compare those purposes or approaches directly.
- Do not compare incidental mentions.
- Prefer passages that explain a paper's method, purpose, approach, contribution, or application.
- Use evidence from every relevant paper.
- Clearly distinguish Paper A from Paper B.
- Do not claim a difference unless the supplied evidence supports both sides.
- Keep the comparison concise.
"""

    prompt = f"""
You are Paper Copilot, a research-paper reading assistant.

Answer the user's question using ONLY the supplied evidence.

Do not use outside knowledge.
Do not guess.
Do not invent information.

{comparison_instruction}

QUESTION

{query}


EVIDENCE

{evidence}


ANSWER RULES

1. Answer ONLY what the user explicitly asked.
2. Use ONLY information supported by the supplied evidence.
3. Do not add unrelated background information.
4. Give the shortest complete answer supported by the evidence.
5. Use bullets when they improve clarity.
6. Cite evidence using ONLY evidence IDs such as:
   [E1]
   [E2]
7. Do NOT write paper names inside citations.
8. Do NOT write page numbers inside citations.
9. Do NOT invent evidence IDs.
10. Only use evidence IDs present in the supplied evidence.
11. Every factual bullet or factual statement must end with at least one evidence ID.
12. If one statement needs multiple passages, use:
    [E1] [E2]
13. For comparison questions, support EACH side of the comparison with its own evidence.
14. Do not add an "Additionally" section unless explicitly requested.
15. Write only cited answer sentences or cited bullets. Omit headings, introductions, repeated questions, and uncited conclusions.
16. For how/method questions, explain the concrete procedure in the evidence, not just its feasibility. When an algorithm is supplied, include its distinct operations in order, including intermediate transformations, aggregation, updates, and any accounting step; do not collapse them into a vague summary.
17. If the evidence is insufficient, respond exactly:
    Not found in the provided evidence.
""".strip()

    return (
        prompt,
        evidence_map,
    )


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Paper Copilot",
    page_icon="📄",
    layout="wide",
)

st.title(
    "📄 Paper Copilot"
)

st.caption(
    "Ask questions across multiple research papers "
    "and verify answers against page-level evidence."
)


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault(
    "pdf_paths",
    [],
)

st.session_state.setdefault(
    "json_paths",
    [],
)

st.session_state.setdefault(
    "idx_dir",
    None,
)

st.session_state.setdefault(
    "results",
    [],
)

# Deep candidate pool used by comparison questions.
st.session_state.setdefault(
    "answer_candidates",
    [],
)

st.session_state.setdefault(
    "answer",
    "",
)

st.session_state.setdefault(
    "view_page",
    None,
)

st.session_state.setdefault(
    "view_document",
    None,
)

st.session_state.setdefault(
    "last_query",
    "",
)

st.session_state.setdefault(
    "last_k",
    0,
)


# ============================================================
# DIRECTORIES
# ============================================================

uploads_dir = Path(
    "data"
)

outputs_dir = Path(
    "outputs"
)

uploads_dir.mkdir(
    exist_ok=True
)

outputs_dir.mkdir(
    exist_ok=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.subheader(
        "📚 Research library"
    )

    pdf_paths = (
        st.session_state.get(
            "pdf_paths",
            [],
        )
        or []
    )

    if pdf_paths:
        st.caption(
            f"{len(pdf_paths)} "
            f"{'paper' if len(pdf_paths) == 1 else 'papers'} "
            "indexed"
        )

        for path in pdf_paths:
            st.write(
                f"• {Path(path).name}"
            )

    else:
        st.info(
            "Upload and process research papers "
            "to create your library."
        )

    st.divider()

    st.subheader(
        "📖 Citation viewer"
    )

    view_page = (
        st.session_state.get(
            "view_page"
        )
    )

    view_document = (
        st.session_state.get(
            "view_document"
        )
    )

    if (
        view_page
        and view_document
    ):
        pdf_path = find_pdf_path(
            view_document
        )

        if pdf_path:
            st.caption(
                f"{Path(view_document).name} · "
                f"Page {view_page}"
            )

            try:
                png = render_page_png(
                    pdf_path,
                    int(
                        view_page
                    ),
                )

                st.image(
                    png,
                    use_container_width=True,
                )

            except Exception as exc:
                st.error(
                    f"Could not render page: {exc}"
                )

        else:
            st.warning(
                "The source PDF could not "
                "be located."
            )

        if st.button(
            "Close page preview",
            use_container_width=True,
        ):
            st.session_state[
                "view_page"
            ] = None

            st.session_state[
                "view_document"
            ] = None

            st.rerun()

    else:
        st.info(
            "Click “View page” on an evidence "
            "card to inspect the original PDF."
        )


# ============================================================
# MULTI-PDF UPLOAD
# ============================================================

pdf_files = st.file_uploader(
    "Upload research papers",
    type=["pdf"],
    accept_multiple_files=True,
)


if pdf_files:
    st.markdown(
        "**Selected papers:**"
    )

    for pdf_file in pdf_files:
        st.write(
            f"• {pdf_file.name}"
        )

    if st.button(
        "Process / re-index library",
        type="primary",
    ):
        saved_pdf_paths: list[
            str
        ] = []

        json_paths: list[
            str
        ] = []

        with st.spinner(
            "Extracting papers and building "
            "the combined research index..."
        ):
            for pdf_file in pdf_files:
                pdf_path = (
                    uploads_dir
                    / pdf_file.name
                )

                pdf_path.write_bytes(
                    pdf_file.getbuffer()
                )

                json_path = (
                    outputs_dir
                    / f"{pdf_path.stem}.json"
                )

                ingest_pdf(
                    pdf_path,
                    json_path,
                    chunk_chars=1200,
                    overlap=200,
                )

                saved_pdf_paths.append(
                    str(pdf_path)
                )

                json_paths.append(
                    str(json_path)
                )

            build_index(
                json_paths,
                LIBRARY_INDEX_DIR,
            )

        st.session_state[
            "pdf_paths"
        ] = saved_pdf_paths

        st.session_state[
            "json_paths"
        ] = json_paths

        st.session_state[
            "idx_dir"
        ] = str(
            LIBRARY_INDEX_DIR
        )

        st.session_state[
            "results"
        ] = []

        st.session_state[
            "answer_candidates"
        ] = []

        st.session_state[
            "answer"
        ] = ""

        st.session_state[
            "view_page"
        ] = None

        st.session_state[
            "view_document"
        ] = None

        st.session_state[
            "last_query"
        ] = ""

        st.session_state[
            "last_k"
        ] = 0

        st.success(
            f"Indexed {len(saved_pdf_paths)} "
            f"{'paper' if len(saved_pdf_paths) == 1 else 'papers'} "
            "into the research library."
        )

        st.rerun()


# ============================================================
# LIBRARY STATUS
# ============================================================

idx_dir = (
    st.session_state.get(
        "idx_dir"
    )
)

if not idx_dir:
    st.info(
        "Upload one or more PDFs and click "
        "“Process / re-index library” to begin."
    )

    st.stop()


st.success(
    f"Research library ready — "
    f"{len(st.session_state['pdf_paths'])} "
    f"{'paper' if len(st.session_state['pdf_paths']) == 1 else 'papers'} "
    "indexed."
)


# ============================================================
# QUESTION
# ============================================================

st.divider()

query = st.text_input(
    "Ask a question about your research library",
    placeholder=(
        "Example: How do the two papers "
        "use machine learning differently?"
    ),
)


# ============================================================
# SETTINGS
# ============================================================

col1, col2 = st.columns(
    [
        1,
        2,
    ]
)

with col1:
    k = st.slider(
        "Evidence passages",
        min_value=3,
        max_value=10,
        value=5,
    )


with col2:
    with st.expander(
        "Advanced settings"
    ):
        retrieval_mode = st.selectbox(
            "Retrieval mode", ["Hybrid", "BM25", "Semantic"],
        )

        ollama_model = (
            st.text_input(
                "Ollama model",
                value="llama3.1:8b",
            )
        )


# ============================================================
# BUTTONS
# ============================================================

button_col1, button_col2, button_col3 = (
    st.columns(3)
)

with button_col1:
    search_clicked = (
        st.button(
            "🔎 Search evidence",
            use_container_width=True,
        )
    )


with button_col2:
    answer_clicked = (
        st.button(
            "✨ Generate answer",
            use_container_width=True,
        )
    )


with button_col3:
    clear_clicked = (
        st.button(
            "Clear",
            use_container_width=True,
        )
    )


# ============================================================
# CLEAR
# ============================================================

if clear_clicked:
    st.session_state[
        "results"
    ] = []

    st.session_state[
        "answer_candidates"
    ] = []

    st.session_state[
        "answer"
    ] = ""

    st.session_state[
        "view_page"
    ] = None

    st.session_state[
        "view_document"
    ] = None

    st.session_state[
        "last_query"
    ] = ""

    st.rerun()


# ============================================================
# SEARCH
# ============================================================

def run_search() -> list[dict]:
    if not query.strip():
        st.warning(
            "Enter a question first."
        )

        return []

    internal_k = max(int(k), COMPARISON_SEARCH_K)

    method = {"BM25": search, "Semantic": semantic_search, "Hybrid": hybrid_search}[retrieval_mode]
    try:
        with st.spinner("Searching the research library..."):
            all_results = method(
                idx_dir, query, k=internal_k, min_score=0.0,
                candidate_multiplier=8, duplicate_threshold=0.82, max_per_page=2,
            )
    except (OSError, ValueError, ImportError) as exc:
        st.session_state["results"] = []
        st.session_state["answer_candidates"] = []
        st.session_state["answer"] = ""
        st.error(f"Search failed: {exc}. Re-index the library or select BM25.")
        return []

    st.session_state["last_retrieval_mode"] = retrieval_mode

    # Only the number chosen in the UI is displayed.
    visible_results = (
        all_results[
            :int(k)
        ]
    )

    st.session_state[
        "results"
    ] = visible_results

    # Answer generation can use the deeper pool.
    st.session_state[
        "answer_candidates"
    ] = all_results

    st.session_state[
        "answer"
    ] = ""

    st.session_state[
        "last_query"
    ] = query

    st.session_state[
        "last_k"
    ] = int(k)

    return visible_results


if search_clicked:
    run_search()


# ============================================================
# GENERATE ANSWER
# ============================================================

if answer_clicked:
    if not query.strip():
        st.warning(
            "Enter a question first."
        )

    else:
        results = (
            st.session_state.get(
                "results",
                [],
            )
            or []
        )

        search_is_current = (
            st.session_state.get(
                "last_query"
            )
            == query
            and st.session_state.get(
                "last_k"
            )
            == int(k)
            and st.session_state.get("last_retrieval_mode") == retrieval_mode
        )

        if (
            not results
            or not search_is_current
        ):
            results = run_search()

        if results:
            candidates = (
                st.session_state.get(
                    "answer_candidates",
                    [],
                )
                or results
            )

            if not evidence_is_sufficient(candidates, query):
                st.session_state[
                    "answer"
                ] = ""

                st.warning(
                    "The retrieved evidence is too weak "
                    "to support a reliable answer."
                )

            else:
                answer_results = (
                    select_answer_evidence(
                        query,
                        candidates,
                    )
                )

                prompt, evidence_map = (
                    build_answer_prompt(
                        query,
                        answer_results,
                    )
                )

                with st.spinner(
                    "Generating a grounded answer..."
                ):
                    try:
                        raw_answer = (
                            ollama_chat(
                                prompt,
                                model=ollama_model,
                            )
                        )

                    except Exception as exc:
                        st.error(
                            f"Ollama error: {exc}"
                        )

                        raw_answer = ""

                if raw_answer:
                    normalized_answer = (
                        normalize_answer(raw_answer, query)
                    )

                    if (
                        normalized_answer
                        == "Not found in the provided evidence."
                    ):
                        st.session_state[
                            "answer"
                        ] = normalized_answer

                    elif evidence_ids_are_valid(
                        normalized_answer,
                        evidence_map,
                    ) and (not is_methodology_question(query)
                           or is_cross_document_question(query)
                           or methodology_answer_is_complete(normalized_answer, answer_results)):
                        final_answer = (
                            replace_evidence_ids_with_citations(
                                normalized_answer,
                                evidence_map,
                            )
                        )

                        st.session_state[
                            "answer"
                        ] = final_answer

                    else:
                        st.warning(
                            "The generated answer used missing "
                            "or invalid citations, uncited text, or omitted algorithm steps, so "
                            "Paper Copilot created a citation-safe "
                            "answer instead."
                        )

                        st.session_state[
                            "answer"
                        ] = (
                            make_extractive_answer(
                                query,
                                answer_results,
                            )
                        )


# ============================================================
# ANSWER DISPLAY
# ============================================================

answer = (
    st.session_state.get(
        "answer",
        "",
    )
)

if answer:
    st.divider()

    st.subheader(
        "Answer"
    )

    st.markdown(
        answer
    )


# ============================================================
# EVIDENCE DISPLAY
# ============================================================

results = (
    st.session_state.get(
        "results",
        [],
    )
    or []
)

if results:
    st.divider()

    st.subheader(
        "Evidence"
    )

    max_coverage = (
        overall_evidence_coverage(
            results
        )
    )

    if not evidence_is_sufficient(
        st.session_state.get("answer_candidates") or results,
        st.session_state.get("last_query", query),
    ):
        st.warning("The retrieved evidence is too weak or lacks the requested entities to support an answer.")
    st.caption(f"Best lexical query coverage: {max_coverage:.0%}")

    if is_cross_document_question(
        st.session_state.get(
            "last_query",
            query,
        )
    ):
        answer_candidates = (
            st.session_state.get(
                "answer_candidates",
                [],
            )
            or results
        )

        retrieved_documents = {
            str(
                result.get(
                    "document",
                    "",
                )
            )
            for result in answer_candidates
        }

        if (
            len(retrieved_documents)
            >= 2
        ):
            st.caption(
                "Cross-paper question detected — "
                f"comparison evidence found across "
                f"{len(retrieved_documents)} papers."
            )

    keywords = query_keywords(
        st.session_state.get(
            "last_query",
            query,
        )
    )

    for result in results:
        rank = int(
            result.get(
                "rank",
                0,
            )
        )

        document = str(
            result.get(
                "document",
                "Unknown paper",
            )
        )

        document_name = (
            document_display_name(
                document
            )
        )

        page = int(
            result.get(
                "page",
                1,
            )
        )

        coverage = float(
            result.get(
                "coverage",
                0.0,
            )
        )

        strength = (
            evidence_strength(
                coverage
            )
        )

        snippet = pick_snippet(
            result.get(
                "text",
                "",
            ),
            keywords,
            query=st.session_state.get(
                "last_query",
                query,
            ),
            max_len=520,
        )

        highlighted = (
            highlight_terms(
                snippet,
                keywords,
            )
        )

        st.markdown(
            f"### #{rank} · "
            f"{document_name} · "
            f"Page {page}"
        )

        meta_col1, meta_col2, meta_col3 = (
            st.columns(
                [
                    1,
                    1,
                    1,
                ]
            )
        )

        with meta_col1:
            st.metric(
                "Lexical match",
                strength,
            )

        with meta_col2:
            st.metric(
                "Query coverage",
                f"{coverage:.0%}",
            )

        with meta_col3:
            if st.button(
                "View page",
                key=(
                    f"view_"
                    f"{rank}_"
                    f"{document_name}_"
                    f"{page}"
                ),
                use_container_width=True,
            ):
                set_view_page(
                    page,
                    document,
                )

                st.rerun()

        st.markdown(
            highlighted,
            unsafe_allow_html=True,
        )

        st.caption(
            "Citation: "
            + citation_text(
                document,
                page,
            )
        )

        with st.expander(
            "Full evidence passage"
        ):
            st.write(
                result.get(
                    "text",
                    "",
                )
            )

        with st.expander(
            "Retrieval details"
        ):
            st.write("Method:", result.get("retrieval_method", "bm25"))
            if "semantic_score" in result:
                st.write("Semantic similarity:", round(result["semantic_score"], 3))
            if "rrf_score" in result:
                st.write("Fusion score:", round(result["rrf_score"], 5))
                st.write("BM25 / semantic ranks:", result["bm25_rank"], result["semantic_rank"])

            st.write(
                "Document:",
                document,
            )

            st.write(
                "Page:",
                page,
            )

            st.write(
                "BM25 score:",
                round(
                    float(
                        result.get(
                            "bm25_score",
                            result.get("score", 0.0) if result.get("retrieval_method") == "bm25" else 0.0,
                        )
                    ),
                    3,
                ),
            )

            st.write(
                "Final rank score:",
                round(
                    float(
                        result.get(
                            "rerank_score",
                            0.0,
                        )
                    ),
                    3,
                ),
            )

            st.write(
                "Query coverage:",
                f"{coverage:.1%}",
            )

            st.write(
                "Matched query terms:",
                result.get(
                    "matched_terms",
                    [],
                ),
            )

            st.write(
                "Intent bonus:",
                round(
                    float(
                        result.get(
                            "intent_bonus",
                            0.0,
                        )
                    ),
                    3,
                ),
            )

            st.write(
                "Noise penalty:",
                round(
                    float(
                        result.get(
                            "noise_penalty",
                            0.0,
                        )
                    ),
                    3,
                ),
            )

        st.divider()