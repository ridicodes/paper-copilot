from pathlib import Path
import re
import fitz
import requests
import streamlit as st

from src.ingest import ingest_pdf
from src.index import build_index, search


BAD_PATTERNS = [
    "issn",
    "international journal",
    "vol.",
    "volume",
    "no.",
    "copyright",
    "all rights reserved",
]


def pick_snippet(text: str, keywords: list[str], max_len: int = 260) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+", t)
    if len(sents) == 1:
        sents = [t[i : i + max_len] for i in range(0, min(len(t), max_len * 3), max_len)]

    def is_bad(s: str) -> bool:
        s_low = s.lower()
        return any(p in s_low for p in BAD_PATTERNS)

    for s in sents:
        s = s.strip()
        if not s or is_bad(s):
            continue
        if any(k in s.lower() for k in keywords):
            return (s[:max_len] + "…") if len(s) > max_len else s

    for s in sents:
        s = s.strip()
        if s and not is_bad(s):
            return (s[:max_len] + "…") if len(s) > max_len else s

    return (t[:max_len] + "…") if len(t) > max_len else t


def make_extractive_answer(query: str, results: list[dict], max_points: int = 6) -> str:
    if not results:
        return "Not found in the paper."

    keywords = [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", query)]
    points = []
    seen = set()

    for r in results:
        page = int(r["page"])
        snippet = pick_snippet(r["text"], keywords)
        key = (page, snippet)
        if key in seen:
            continue
        seen.add(key)
        points.append(f"- {snippet} (p{page})")
        if len(points) >= max_points:
            break

    return "\n".join(points) if points else "Not found in the paper."


@st.cache_data
def render_page_png(pdf_path: str, page_num: int, zoom: float = 1.6) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def set_view_page(page: int):
    st.session_state["view_page"] = int(page)


def ollama_answer(query: str, results: list[dict], model: str) -> str:
    keywords = [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", query)]
    evidence_blocks = []
    for r in results:
        p = int(r["page"])
        snip = pick_snippet(r["text"], keywords, max_len=520)
        evidence_blocks.append(f"[p{p}] {snip}")
    evidence = "\n".join(evidence_blocks)

    prompt = f"""
You are a paper reading assistant.
Use ONLY the evidence provided. If the evidence does not contain the answer, say "Not found in the provided evidence."

Task:
Answer the question in 4 to 7 bullet points.

Hard rules:
- Every bullet MUST end with one or more page citations like (p5) or (p2, p7).
- Do not invent datasets, numbers, or claims.

Question:
{query}

Evidence:
{evidence}
""".strip()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


st.set_page_config(page_title="Paper Copilot", layout="wide")
st.title("Paper Copilot")
st.write("Upload a research paper PDF, then search with page citations.")

# session state defaults
st.session_state.setdefault("pdf_path", None)
st.session_state.setdefault("idx_dir", None)
st.session_state.setdefault("results", [])
st.session_state.setdefault("answer", "")
st.session_state.setdefault("view_page", None)
st.session_state.setdefault("last_query", "")
st.session_state.setdefault("last_k", 0)

uploads_dir = Path("data")
outputs_dir = Path("outputs")
uploads_dir.mkdir(exist_ok=True)
outputs_dir.mkdir(exist_ok=True)

# sidebar preview (small and stable)
with st.sidebar:
    st.subheader("Page preview")
    if st.session_state.get("pdf_path") and st.session_state.get("view_page"):
        vp = int(st.session_state["view_page"])
        st.caption(f"Page {vp}")
        png = render_page_png(st.session_state["pdf_path"], vp)
        st.image(png, width=280)
        if st.button("Clear preview"):
            st.session_state["view_page"] = None
    else:
        st.caption("Click a View page button in Evidence.")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    pdf_path = uploads_dir / pdf_file.name
    pdf_path.write_bytes(pdf_file.getbuffer())
    st.success(f"Saved: {pdf_path.name}")

    if st.button("Process PDF"):
        json_path = outputs_dir / (pdf_path.stem + ".json")
        ingest_pdf(pdf_path, json_path)

        idx_dir = outputs_dir / pdf_path.stem
        build_index(json_path, idx_dir)

        st.session_state["pdf_path"] = str(pdf_path)
        st.session_state["idx_dir"] = str(idx_dir)
        st.session_state["results"] = []
        st.session_state["answer"] = ""
        st.session_state["view_page"] = None

        st.success("Processed and indexed")

query = st.text_input("Ask a question", value=st.session_state.get("last_query", ""))
k = st.slider("Evidence chunks", 3, 15, 5)

model = st.text_input("Ollama model", value="llama3.1:8b")

# if query or k changed, clear old outputs so you do not see stale stuff
if query != st.session_state.get("last_query") or int(k) != int(st.session_state.get("last_k") or 0):
    st.session_state["results"] = []
    st.session_state["answer"] = ""
    st.session_state["last_query"] = query
    st.session_state["last_k"] = int(k)

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    do_search = st.button("Search evidence")
with colB:
    do_answer = st.button("Generate answer")
with colC:
    if st.button("Clear results"):
        st.session_state["results"] = []
        st.session_state["answer"] = ""
        st.session_state["view_page"] = None

idx_dir = st.session_state.get("idx_dir")

def run_search():
    if not idx_dir:
        st.error("Click Process PDF first.")
        return
    if not query.strip():
        st.warning("Type a question first.")
        return
    st.session_state["results"] = search(idx_dir, query, k=int(k))

if do_search:
    run_search()

if do_answer:
    # always ensure results exist
    if not st.session_state.get("results"):
        run_search()

    results = st.session_state.get("results") or []
    if results:
        try:
            ans = ollama_answer(query, results, model=model)
        except Exception as e:
            st.session_state["answer"] = ""
            st.error(f"Ollama call failed: {e}")
        else:
            # if model ignored citations, fallback to extractive answer
            if "(p" not in ans:
                st.warning("Model did not include page citations, showing extractive answer instead.")
                ans = make_extractive_answer(query, results)
            st.session_state["answer"] = ans

results = st.session_state.get("results") or []
if results:
    st.subheader(f"Evidence (showing {len(results)} results)")
    for r in results:
        page = int(r["page"])
        st.markdown(f"Rank {r['rank']} | Page {page} | score {r['score']:.3f}")

        # show a short snippet, keep UI clean
        keywords = [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", query)]
        st.write(pick_snippet(r["text"], keywords, max_len=320))

        st.button("View page", key=f"view_{r['rank']}_{page}", on_click=set_view_page, args=(page,))

        with st.expander("Show full chunk"):
            st.write(r["text"])

        st.divider()

ans = st.session_state.get("answer") or ""
if ans:
    st.subheader("Answer (with citations)")
    st.markdown(ans)
