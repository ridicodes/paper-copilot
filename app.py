from pathlib import Path
import re

import fitz
import streamlit as st

from src.ingest import ingest_pdf
from src.index import build_index, search
from src.llm import ollama_chat


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

BAD_PATTERNS = [
    "issn",
    "international journal",
    "copyright",
    "all rights reserved",
    "no researchers usefulness definition",
]


# ---------------------------------------------------------
# Query processing
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()

    if not cleaned:
        return []

    # Also split around numbered list markers such as
    # "1)", "2)", etc.
    parts = re.split(
        r"(?<=[.!?])\s+|(?=\b\d+\)\s*)",
        cleaned,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# ---------------------------------------------------------
# Evidence snippet selection
# ---------------------------------------------------------

def pick_snippet(
    text: str,
    keywords: list[str],
    query: str = "",
    max_len: int = 360,
) -> str:
    """
    Return the most query-relevant window from a retrieved chunk.

    Week 3 improvements:
    - score sentences using matched query terms,
    - reward exact intent words,
    - surface useful numbered lists,
    - avoid obvious publication/reference noise.
    """

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

        # Reward query-term matches.
        for keyword in keywords:
            if keyword in low:
                score += 1.0

        # Intent-aware snippet bonuses.
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

        if score > best_score:
            best_score = score
            best_index = i

    # Start with the strongest sentence.
    chosen = [sentences[best_index]]

    # For process/list questions, include subsequent
    # numbered items when possible.
    if any(
        word in q_low
        for word in [
            "stages",
            "steps",
            "phases",
            "process",
        ]
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
                chosen.append(sentences[j])
                j += 1
            else:
                break

    snippet = " ".join(chosen).strip()

    if len(snippet) > max_len:
        snippet = (
            snippet[:max_len].rstrip()
            + "…"
        )

    return snippet


# ---------------------------------------------------------
# Extractive fallback answer
# ---------------------------------------------------------

def make_extractive_answer(
    query: str,
    results: list[dict],
    max_points: int = 6,
) -> str:
    if not results:
        return "Not found in the paper."

    keywords = query_keywords(query)

    points: list[str] = []
    seen: set[tuple[int, str]] = set()

    for result in results:
        page = int(result["page"])

        snippet = pick_snippet(
            result["text"],
            keywords,
            query=query,
            max_len=420,
        )

        key = (page, snippet)

        if key in seen:
            continue

        seen.add(key)

        points.append(
            f"- {snippet} (p{page})"
        )

        if len(points) >= max_points:
            break

    if not points:
        return "Not found in the paper."

    return "\n".join(points)


# ---------------------------------------------------------
# Week 3: insufficient-evidence detection
# ---------------------------------------------------------

def evidence_is_sufficient(
    results: list[dict],
) -> tuple[bool, str]:
    """
    Decide whether retrieved evidence is strong enough
    to allow answer generation.

    Week 3 rule:
    - at least one result must exist,
    - the best result must cover at least 50% of the
      meaningful query terms.

    Query coverage is used instead of raw BM25 because
    BM25 scores are not directly comparable between
    different questions.
    """

    if not results:
        return (
            False,
            "No evidence was retrieved.",
        )

    best_coverage = max(
        float(
            result.get(
                "coverage",
                0.0,
            )
        )
        for result in results
    )

    if best_coverage < 0.50:
        return (
            False,
            (
                "Best retrieved evidence covers only "
                f"{best_coverage:.0%} of the important "
                "query terms."
            ),
        )

    return True, ""


# ---------------------------------------------------------
# PDF page rendering
# ---------------------------------------------------------

@st.cache_data
def render_page_png(
    pdf_path: str,
    page_num: int,
    zoom: float = 1.6,
) -> bytes:
    doc = fitz.open(pdf_path)

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

        return pix.tobytes("png")

    finally:
        doc.close()


def set_view_page(page: int) -> None:
    st.session_state["view_page"] = int(page)


# ---------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------

def build_answer_prompt(
    query: str,
    results: list[dict],
) -> str:
    evidence_blocks: list[str] = []

    for result in results:
        page = int(result["page"])

        text = re.sub(
            r"\s+",
            " ",
            result["text"],
        ).strip()

        evidence_blocks.append(
            f"[p{page}] {text}"
        )

    evidence = "\n\n".join(
        evidence_blocks
    )

    return f"""
You are Paper Copilot, a research-paper reading assistant.

Use ONLY the evidence below.
Do not use outside knowledge.

If the evidence does not contain enough information to answer the question,
say "Not found in the provided evidence."

Question:
{query}

Evidence:
{evidence}

Answer requirements:
- Give a concise, direct answer in 3 to 7 bullet points.
- Every factual bullet MUST end with one or more page citations such as
  (p5) or (p2, p7).
- Use only page numbers that appear in the evidence.
- Do not invent datasets, methods, numbers, conclusions, or citations.
""".strip()


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Paper Copilot",
    page_icon="📄",
    layout="wide",
)

st.title("Paper Copilot")

st.write(
    "Upload a research paper PDF, retrieve the strongest evidence, "
    "and generate a page-cited answer."
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------

uploads_dir = Path("data")
outputs_dir = Path("outputs")

uploads_dir.mkdir(
    exist_ok=True
)

outputs_dir.mkdir(
    exist_ok=True
)


# ---------------------------------------------------------
# Sidebar page preview
# ---------------------------------------------------------

with st.sidebar:
    st.subheader(
        "Page preview"
    )

    if (
        st.session_state.get("pdf_path")
        and st.session_state.get("view_page")
    ):
        view_page = int(
            st.session_state["view_page"]
        )

        st.caption(
            f"Page {view_page}"
        )

        try:
            png = render_page_png(
                st.session_state["pdf_path"],
                view_page,
            )

            st.image(
                png,
                width=280,
            )

        except Exception as exc:
            st.error(
                f"Could not render page: {exc}"
            )

        if st.button(
            "Clear preview"
        ):
            st.session_state[
                "view_page"
            ] = None

            st.rerun()

    else:
        st.caption(
            "Click a View page button in Evidence."
        )


# ---------------------------------------------------------
# PDF upload
# ---------------------------------------------------------

pdf_file = st.file_uploader(
    "Upload a PDF",
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
        ] = str(pdf_path)

        st.session_state[
            "idx_dir"
        ] = str(idx_dir)

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


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

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
    "Evidence chunks",
    min_value=3,
    max_value=15,
    value=5,
)


model = st.text_input(
    "Ollama model",
    value="llama3.1:8b",
)


# ---------------------------------------------------------
# Reset old results when question changes
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Buttons
# ---------------------------------------------------------

col_a, col_b, col_c = st.columns(
    [1, 1, 1]
)

with col_a:
    do_search = st.button(
        "Search evidence",
        use_container_width=True,
    )

with col_b:
    do_answer = st.button(
        "Generate answer",
        use_container_width=True,
    )

with col_c:
    clear_results = st.button(
        "Clear results",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Clear results
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

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
            "Please click Process / re-index PDF again."
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
            "No BM25 evidence was found for this question. "
            "Try different wording or a more specific question."
        )
        return False

    return True


# ---------------------------------------------------------
# Search button
# ---------------------------------------------------------

if do_search:
    run_search()


# ---------------------------------------------------------
# Generate grounded answer
# ---------------------------------------------------------

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

    sufficient, reason = (
        evidence_is_sufficient(
            results
        )
    )

    # Week 3 abstention:
    # Do NOT send weak evidence to Ollama.
    if results and not sufficient:
        st.session_state[
            "answer"
        ] = ""

        st.warning(
            "⚠️ Insufficient evidence in the uploaded paper. "
            + reason
            + " Paper Copilot will not generate an answer "
              "from weak evidence."
        )

    elif results:
        prompt = build_answer_prompt(
            query,
            results,
        )

        with st.spinner(
            "Generating a grounded answer with Ollama..."
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

            st.error(answer)

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

            st.error(answer)

        else:
            # If the model fails to include citations,
            # fall back to an extractive grounded answer.
            if "(p" not in answer:
                st.warning(
                    "The model omitted page citations, "
                    "so Paper Copilot is showing a "
                    "grounded extractive answer instead."
                )

                answer = (
                    make_extractive_answer(
                        query,
                        results,
                    )
                )

            st.session_state[
                "answer"
            ] = answer


# ---------------------------------------------------------
# Evidence display
# ---------------------------------------------------------

results = (
    st.session_state.get(
        "results"
    )
    or []
)

if results:
    st.subheader(
        f"Evidence ({len(results)} results)"
    )

    keywords = query_keywords(
        query
    )

    # Week 3 evidence confidence check.
    sufficient, reason = (
        evidence_is_sufficient(
            results
        )
    )

    if not sufficient:
        st.warning(
            "⚠️ Insufficient evidence: "
            + reason
            + " These are the closest passages found, "
              "but they may not answer the question."
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

        st.markdown(
            f"**Rank {result['rank']} · "
            f"Page {page}**  \n"
            f"BM25: `{bm25_score:.3f}` · "
            f"final rank score: "
            f"`{rerank_score:.3f}` · "
            f"query coverage: "
            f"`{coverage:.0%}`"
        )

        details = []

        if matched_terms:
            details.append(
                "matched: "
                + ", ".join(
                    matched_terms
                )
            )

        if intent_bonus > 0:
            details.append(
                f"intent bonus: "
                f"+{intent_bonus:.1f}"
            )

        if noise_penalty > 0:
            details.append(
                f"noise penalty: "
                f"-{noise_penalty:.1f}"
            )

        if details:
            st.caption(
                " · ".join(details)
            )

        st.write(
            pick_snippet(
                result["text"],
                keywords,
                query=query,
                max_len=420,
            )
        )

        st.button(
            "View page",
            key=(
                f"view_"
                f"{result['rank']}_"
                f"{page}_"
                f"{result.get('chunk_id', '')}"
            ),
            on_click=set_view_page,
            args=(page,),
        )

        with st.expander(
            "Show full chunk"
        ):
            st.write(
                result["text"]
            )

        st.divider()


# ---------------------------------------------------------
# Answer display
# ---------------------------------------------------------

answer = (
    st.session_state.get(
        "answer"
    )
    or ""
)

if answer:
    st.subheader(
        "Answer (with citations)"
    )

    st.markdown(answer)