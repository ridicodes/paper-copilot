# src/ingest.py
import json
import re
from pathlib import Path
import fitz  # PyMuPDF


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    out_json: str | Path,
    chunk_chars: int = 1200,
    overlap: int = 200,
) -> Path:
    pdf_path = Path(pdf_path)
    out_json = Path(out_json)

    doc = fitz.open(str(pdf_path))
    chunks = []

    for i in range(len(doc)):
        page = doc.load_page(i)
        raw = page.get_text("text")
        text = _clean_text(raw)
        if not text:
            continue

        page_chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
        for j, ch in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "page": i + 1,
                    "chunk_id": f"{i+1}-{j}",
                    "text": ch,
                }
            )

    payload = {
        "source": pdf_path.name,
        "num_pages": len(doc),
        "chunk_chars": chunk_chars,
        "overlap": overlap,
        "chunks": chunks,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json
