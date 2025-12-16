from src.ingest import ingest_pdf
from src.index import build_index, search
from pathlib import Path
import streamlit as st

def make_answer(query: str, results: list[dict], max_points: int = 6) -> str:
    if not results:
        return "Not found in the paper."

    points = []
    for r in results[:max_points]:
        page = r["page"]
        text = r["text"].strip()

        # Take first 1–2 sentences as a compact evidence-based point
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        snippet = ". ".join(sentences[:2]).strip()
        if snippet and not snippet.endswith("."):
            snippet += "."

        points.append(f"- {snippet} (p{page})")

    return "\n".join(points)


st.set_page_config(page_title="Paper Copilot", layout="wide")
st.title("Paper Copilot")
st.write("Upload a research paper PDF, then search with page citations.")

uploads_dir = Path("data")
outputs_dir = Path("outputs")
uploads_dir.mkdir(exist_ok=True)
outputs_dir.mkdir(exist_ok=True)

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

        st.session_state["idx_dir"] = str(idx_dir)
        st.success("Processed and indexed")

query = st.text_input("Ask: What is the main contribution? What dataset did they use?")
k = st.slider("Evidence chunks", 3, 10, 5)

idx_dir = st.session_state.get("idx_dir")

colA, colB = st.columns([1, 1])
with colA:
    do_search = st.button("Search evidence")
with colB:
    do_answer = st.button("Generate answer")

if (do_search or do_answer) and query:
    if not idx_dir:
        st.error("Click Process PDF first.")
    else:
        results = search(idx_dir, query, k=k)

        st.subheader("Evidence")
        for r in results:
            st.markdown(f"**{r['rank']}. Page {r['page']}** (score {r['score']:.3f})")
            st.write(r["text"])
            st.divider()

        if do_answer:
            st.subheader("Answer (with citations)")
            st.markdown(make_answer(query, results))



