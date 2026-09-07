from pathlib import Path
import re
import html

import fitz
import streamlit as st

from src.ingest import ingest_pdf
from src.index import build_index, search
from src.llm import ollama_chat


# ============================================================
# WEEK 4 CONFIGURATION
# ============================================================

BAD_PATTERNS = [
    "issn",
    "international journal",
    "copyright",
    "all rights reserved",
    "no researchers usefulness definition",
]

# Minimum query coverage required before allowing an LLM answer.
MIN_ANSWER_COVERAGE = 0.50

# Maximum number of evidence passages sent to Ollama.
MAX_ANSWER_PASSAGES = 3


# ============================================================
# QUERY / TEXT HELPERS
# ============================================================

def query_keywords(query: str) -> list[str]:
    generic = {
        "paper",
        "study",
        "article",
        "described",
        "discussed",
        "explain",
        "explains",
        "main",
        "according",
        "authors",
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

        # Intent-aware snippet bonuses.
        if (
            "stages" in q_low
            and "stages" in low
        ):
            score += 4.0

        if (
            "steps" in q_low
            and "steps" in low
        ):
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

        if score > best_score:
            best_score = score
            best_index = i

    chosen = [
        sentences[best_index]
    ]

    # Preserve useful numbered sequences for process questions.
    if any(
    word in q_low
    for word in [
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
    ]
):
        j = best_index + 1

        while (
            j < len(sentences)
            and len(
                " ".join(chosen)
            ) < max_len
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

    snippet = (
        " ".join(chosen)
        .strip()
    )

    if len(snippet) > max_len:
        snippet = (
            snippet[:max_len]
            .rstrip()
            + "…"
        )

    return snippet


# ============================================================
# WEEK 4: EVIDENCE PRESENTATION
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


def evidence_is_sufficient(
    results: list[dict],
) -> bool:

    return (
        overall_evidence_coverage(
            results
        )
        >= MIN_ANSWER_COVERAGE
    )


# ============================================================
# WEEK 4: SELECT EVIDENCE FOR ANSWER GENERATION
# ============================================================

def get_matched_query_terms(
    result: dict,
    keywords: list[str],
) -> set[str]:
    """
    Return important query terms covered by one evidence passage.

    Uses the retriever's matched_terms when available and also
    checks the passage text directly.
    """

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


def select_answer_evidence(
    query: str,
    results: list[dict],
    max_passages: int = MAX_ANSWER_PASSAGES,
) -> list[dict]:
    """
    Select focused evidence for answer generation.

    For list-style questions whose answer may span multiple
    passages, keep a few of the strongest results.

    For ordinary questions, use the minimum evidence needed
    to cover the important query terms.
    """

    if not results:
        return []

    keywords = query_keywords(query)
    q_low = query.lower()

    # Questions whose answers are commonly spread over
    # multiple chunks/pages.
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

        for result in results[:max_passages]:

            # Keep the strongest result.
            if not selected:
                selected.append(result)
                continue

            # For supporting passages, allow weaker lexical
            # coverage because they may continue an answer
            # introduced in the top passage.
            coverage = float(
                result.get(
                    "coverage",
                    0.0,
                )
            )

            if coverage >= 0.20:
                selected.append(result)

        return selected

    # --------------------------------------------------------
    # Normal focused-answer behaviour
    # --------------------------------------------------------

    required_terms = set(keywords)

    selected: list[dict] = []
    covered_terms: set[str] = set()

    for result in results:

        coverage = float(
            result.get(
                "coverage",
                0.0,
            )
        )

        if coverage < MIN_ANSWER_COVERAGE:
            continue

        result_terms = get_matched_query_terms(
            result,
            keywords,
        )

        if not selected:

            selected.append(result)
            covered_terms.update(result_terms)

        else:

            new_terms = (
                result_terms
                - covered_terms
            )

            if new_terms:

                selected.append(result)
                covered_terms.update(result_terms)

        if (
            required_terms
            and required_terms.issubset(
                covered_terms
            )
        ):
            break

        if len(selected) >= max_passages:
            break

    if not selected:
        selected = [results[0]]

    return selected

# ============================================================
# EXTRACTIVE FALLBACK ANSWER
# ============================================================

def make_extractive_answer(
    query: str,
    results: list[dict],
    max_points: int = 6,
) -> str:

    if not results:
        return "Not found in the paper."

    keywords = query_keywords(
        query
    )

    points: list[str] = []

    seen: set[
        tuple[int, str]
    ] = set()

    for result in results:

        page = int(
            result["page"]
        )

        snippet = pick_snippet(
            result["text"],
            keywords,
            query=query,
            max_len=420,
        )

        key = (
            page,
            snippet,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        points.append(
            f"- {snippet} **[p. {page}]**"
        )

        if (
            len(points)
            >= max_points
        ):
            break

    if not points:
        return "Not found in the paper."

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
) -> None:

    st.session_state[
        "view_page"
    ] = int(page)


# ============================================================
# OLLAMA PROMPT
# ============================================================

def build_answer_prompt(
    query: str,
    results: list[dict],
) -> str:

    evidence_blocks: list[str] = []

    for result in results:

        page = int(
            result["page"]
        )

        text = pick_snippet(

    result["text"],

    query_keywords(query),

    query=query,

    max_len=900,

)

        evidence_blocks.append(
            f"[p. {page}] {text}"
        )

    evidence = "\n\n".join(
        evidence_blocks
    )

    return f"""
You are Paper Copilot, a research-paper reading assistant.

Answer the user's question using ONLY the supplied evidence.

Do not use outside knowledge.
Do not guess.
Do not invent information.

If the supplied evidence does not support a claim, do not include that claim.

QUESTION

{query}


EVIDENCE

{evidence}


ANSWER RULES

1. Answer ONLY what the user explicitly asked.
2. Do not add related, supplementary, background, or follow-up information.
3. Ignore evidence that does not directly help answer the exact question.
4. Give the shortest complete answer supported by the evidence.
5. Use bullets or a numbered list when the question asks for stages, steps, items, weaknesses, advantages, or other lists.
6. Every factual bullet or factual statement must end with a page citation.
7. Format citations exactly like:
   [p. 2]
   [p. 2, p. 4]
8. Only cite pages present in the supplied evidence.
9. Never invent a page number.
10. Never cite outside sources.
11. Do not repeat the question unless necessary.
12. Do not add an "Additionally" section or discuss related concepts unless the user asks for them.
13. If the evidence is insufficient, respond exactly:
    Not found in the provided evidence.
""".strip()


# ============================================================
# CITATION VALIDATION
# ============================================================

def answer_has_citations(
    answer: str,
) -> bool:

    return bool(
        re.search(
            r"\[p\.\s*\d+",
            answer,
            flags=re.IGNORECASE,
        )
    )


def cited_pages(
    answer: str,
) -> set[int]:
    """
    Extract page numbers from citations such as:
    [p. 2]
    [p. 2, p. 4]
    """

    pages = re.findall(
        r"p\.\s*(\d+)",
        answer,
        flags=re.IGNORECASE,
    )

    return {
        int(page)
        for page in pages
    }


def citations_are_valid(
    answer: str,
    evidence_results: list[dict],
) -> bool:

    allowed_pages = {
        int(result["page"])
        for result in evidence_results
    }

    used_pages = cited_pages(
        answer
    )

    if not used_pages:
        return False

    return used_pages.issubset(
        allowed_pages
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
    "Ask questions about a research paper and verify "
    "answers directly against page-level evidence."
)


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault(
    "pdf_path",
    None,
)

st.session_state.setdefault(
    "idx_dir",
    None,
)

st.session_state.setdefault(
    "results",
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
# SIDEBAR PAGE PREVIEW
# ============================================================

with st.sidebar:

    st.subheader(
        "📖 Citation viewer"
    )

    if (
        st.session_state.get(
            "pdf_path"
        )
        and st.session_state.get(
            "view_page"
        )
    ):

        view_page = int(
            st.session_state[
                "view_page"
            ]
        )

        st.caption(
            f"Viewing cited page {view_page}"
        )

        try:

            png = render_page_png(
                st.session_state[
                    "pdf_path"
                ],
                view_page,
            )

            st.image(
                png,
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                f"Could not render page: {exc}"
            )

        if st.button(
            "Close page preview",
            use_container_width=True,
        ):

            st.session_state[
                "view_page"
            ] = None

            st.rerun()

    else:

        st.info(
            "Select **View page** beside an evidence "
            "passage to verify it against the PDF."
        )


# ============================================================
# PDF UPLOAD
# ============================================================

pdf_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"],
)

if pdf_file:

    pdf_path = (
        uploads_dir
        / pdf_file.name
    )

    pdf_path.write_bytes(
        pdf_file.getbuffer()
    )

    st.success(
        f"Loaded: {pdf_path.name}"
    )

    if st.button(
        "Process / re-index PDF"
    ):

        with st.spinner(
            "Extracting text and rebuilding "
            "the retrieval index..."
        ):

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

            idx_dir = (
                outputs_dir
                / pdf_path.stem
            )

            build_index(
                json_path,
                idx_dir,
            )

        st.session_state[
            "pdf_path"
        ] = str(
            pdf_path
        )

        st.session_state[
            "idx_dir"
        ] = str(
            idx_dir
        )

        st.session_state[
            "results"
        ] = []

        st.session_state[
            "answer"
        ] = ""

        st.session_state[
            "view_page"
        ] = None

        st.session_state[
            "last_query"
        ] = ""

        st.session_state[
            "last_k"
        ] = 0

        st.success(
            "Processed and indexed."
        )


# ============================================================
# QUESTION
# ============================================================

query = st.text_input(
    "Ask a question",
    value=st.session_state.get(
        "last_query",
        "",
    ),
    placeholder=(
        "e.g. What are the main stages "
        "of image analysis?"
    ),
)


k = st.slider(
    "Evidence passages",
    min_value=3,
    max_value=10,
    value=5,
)


with st.expander(
    "Advanced settings"
):

    model = st.text_input(
        "Ollama model",
        value="llama3.1:8b",
    )


# ============================================================
# RESET RESULTS WHEN QUESTION CHANGES
# ============================================================

if (
    query
    != st.session_state.get(
        "last_query"
    )
    or int(k)
    != int(
        st.session_state.get(
            "last_k"
        )
        or 0
    )
):

    st.session_state[
        "results"
    ] = []

    st.session_state[
        "answer"
    ] = ""

    st.session_state[
        "last_query"
    ] = query

    st.session_state[
        "last_k"
    ] = int(k)


# ============================================================
# BUTTONS
# ============================================================

col_a, col_b, col_c = st.columns(
    [1, 1, 1]
)

with col_a:

    do_search = st.button(
        "🔎 Search evidence",
        use_container_width=True,
    )

with col_b:

    do_answer = st.button(
        "✨ Generate answer",
        use_container_width=True,
    )

with col_c:

    clear_results = st.button(
        "Clear",
        use_container_width=True,
    )


if clear_results:

    st.session_state[
        "results"
    ] = []

    st.session_state[
        "answer"
    ] = ""

    st.session_state[
        "view_page"
    ] = None

    st.rerun()


idx_dir = st.session_state.get(
    "idx_dir"
)


# ============================================================
# SEARCH
# ============================================================

def run_search() -> bool:

    if not idx_dir:

        st.error(
            "Upload a PDF and click "
            "Process / re-index PDF first."
        )

        return False

    if not query.strip():

        st.warning(
            "Type a question first."
        )

        return False

    try:

        results = search(
            idx_dir,
            query,
            k=int(k),
            min_score=0.0,
            candidate_multiplier=8,
            duplicate_threshold=0.82,
            max_per_page=2,
        )

    except FileNotFoundError:

        st.error(
            "The index could not be found. "
            "Please process the PDF again."
        )

        return False

    except Exception as exc:

        st.error(
            f"Search failed: {exc}"
        )

        return False

    st.session_state[
        "results"
    ] = results

    if not results:

        st.info(
            "No useful evidence was found. "
            "Try different wording or a "
            "more specific question."
        )

        return False

    return True


if do_search:
    run_search()


# ============================================================
# ANSWER GENERATION
# ============================================================

if do_answer:

    if not st.session_state.get(
        "results"
    ):
        run_search()

    results = (
        st.session_state.get(
            "results"
        )
        or []
    )

    if results:

        coverage = (
            overall_evidence_coverage(
                results
            )
        )

        # Do not send weak evidence to the LLM.
        if not evidence_is_sufficient(
            results
        ):

            st.session_state[
                "answer"
            ] = ""

            st.warning(
                "I found related passages, but the "
                f"best evidence covers only "
                f"{coverage:.0%} of the important "
                "query terms. Paper Copilot will "
                "not generate an answer from weak "
                "evidence."
            )

        else:

            # WEEK 4:
            # Keep all results visible to the user,
            # but send only the minimum necessary
            # evidence to Ollama.
            answer_results = (
                select_answer_evidence(
                    query,
                    results,
                )
            )

            prompt = build_answer_prompt(
                query,
                answer_results,
            )

            with st.spinner(
                "Generating a grounded answer..."
            ):

                answer = ollama_chat(
                    prompt,
                    model=model,
                )

            if answer.startswith(
                "Could not connect to Ollama"
            ):

                st.session_state[
                    "answer"
                ] = ""

                st.error(
                    answer
                )

            elif (
                answer.startswith(
                    "Ollama error"
                )
                or answer.startswith(
                    "Ollama timed out"
                )
            ):

                st.session_state[
                    "answer"
                ] = ""

                st.error(
                    answer
                )

            else:

                # Citation safeguard:
                # the model must include citations
                # and may cite only evidence pages
                # actually supplied to it.
                if (
                    not answer_has_citations(
                        answer
                    )
                    or not citations_are_valid(
                        answer,
                        answer_results,
                    )
                ):

                    st.warning(
                        "The model produced missing or "
                        "invalid page citations. Showing "
                        "a citation-safe extractive answer "
                        "instead."
                    )

                    answer = (
                        make_extractive_answer(
                            query,
                            answer_results,
                        )
                    )

                st.session_state[
                    "answer"
                ] = answer


# ============================================================
# EVIDENCE DISPLAY
# ============================================================

results = (
    st.session_state.get(
        "results"
    )
    or []
)

if results:

    st.divider()

    st.subheader(
        f"Evidence · {len(results)} passages"
    )

    best_coverage = (
        overall_evidence_coverage(
            results
        )
    )

    if evidence_is_sufficient(
        results
    ):

        st.success(
            f"{evidence_strength(best_coverage)} "
            f"evidence match · "
            f"{best_coverage:.0%} query coverage"
        )

    else:

        st.warning(
            "Insufficient evidence · "
            f"best passage covers "
            f"{best_coverage:.0%} of the "
            "important query terms."
        )

    keywords = query_keywords(
        query
    )

    for result in results:

        page = int(
            result["page"]
        )

        bm25_score = float(
            result["score"]
        )

        rerank_score = float(
            result.get(
                "rerank_score",
                bm25_score,
            )
        )

        coverage = float(
            result.get(
                "coverage",
                0.0,
            )
        )

        matched_terms = (
            result.get(
                "matched_terms",
                [],
            )
        )

        intent_bonus = float(
            result.get(
                "intent_bonus",
                0.0,
            )
        )

        noise_penalty = float(
            result.get(
                "noise_penalty",
                0.0,
            )
        )

        # ----------------------------
        # Evidence card heading
        # ----------------------------

        st.markdown(
            f"### Evidence {result['rank']} "
            f"· Page {page}"
        )

        strength = (
            evidence_strength(
                coverage
            )
        )

        st.caption(
            f"{strength} match · "
            f"{coverage:.0%} query coverage"
        )

        snippet = pick_snippet(
            result["text"],
            keywords,
            query=query,
            max_len=420,
        )

        highlighted = (
            highlight_terms(
                snippet,
                matched_terms,
            )
        )

        st.markdown(
            highlighted,
            unsafe_allow_html=True,
        )

        # ----------------------------
        # Citation / page verification
        # ----------------------------

        left, right = st.columns(
            [1, 5]
        )

        with left:

            st.button(
                f"📖 View page {page}",
                key=(
                    f"view_"
                    f"{result['rank']}_"
                    f"{page}_"
                    f"{result.get('chunk_id', '')}"
                ),
                on_click=set_view_page,
                args=(page,),
                use_container_width=True,
            )

        with right:

            st.markdown(
                f"**Citation:** `[p. {page}]`"
            )

        # ----------------------------
        # Full evidence
        # ----------------------------

        with st.expander(
            "Show full evidence passage"
        ):

            st.write(
                result["text"]
            )

        # ----------------------------
        # Technical retrieval details
        # ----------------------------

        with st.expander(
            "Retrieval details"
        ):

            st.write(
                f"BM25 score: "
                f"{bm25_score:.3f}"
            )

            st.write(
                f"Final rerank score: "
                f"{rerank_score:.3f}"
            )

            st.write(
                f"Query coverage: "
                f"{coverage:.0%}"
            )

            if matched_terms:

                st.write(
                    "Matched terms: "
                    + ", ".join(
                        matched_terms
                    )
                )

            if intent_bonus > 0:

                st.write(
                    f"Intent bonus: "
                    f"+{intent_bonus:.1f}"
                )

            if noise_penalty > 0:

                st.write(
                    f"Noise penalty: "
                    f"-{noise_penalty:.1f}"
                )

        st.divider()


# ============================================================
# ANSWER DISPLAY
# ============================================================

answer = (
    st.session_state.get(
        "answer"
    )
    or ""
)

if answer:

    st.subheader(
        "Answer"
    )

    st.markdown(
        answer
    )

    st.caption(
        "Answers are generated only from the "
        "retrieved passages above. Verify claims "
        "using the cited PDF pages."
    )