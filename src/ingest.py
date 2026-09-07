import json
import re
from pathlib import Path

import fitz

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text]


def _chunk_text(text: str, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        if len(sentence) > chunk_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []

            step = max(1, chunk_chars - overlap_chars)
            start = 0
            while start < len(sentence):
                piece = sentence[start:start + chunk_chars].strip()
                if piece:
                    chunks.append(piece)
                if start + chunk_chars >= len(sentence):
                    break
                start += step
            continue

        proposed = " ".join(current + [sentence])
        if current and len(proposed) > chunk_chars:
            finished = " ".join(current).strip()
            if finished:
                chunks.append(finished)

            overlap_sentences: list[str] = []
            overlap_len = 0
            for old_sentence in reversed(current):
                additional = len(old_sentence) + (1 if overlap_sentences else 0)
                if overlap_sentences and overlap_len + additional > overlap_chars:
                    break
                overlap_sentences.insert(0, old_sentence)
                overlap_len += additional
            current = overlap_sentences

        current.append(sentence)

    if current:
        final_chunk = " ".join(current).strip()
        if final_chunk and (not chunks or final_chunk != chunks[-1]):
            chunks.append(final_chunk)

    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    out_json: str | Path,
    chunk_chars: int = 1200,
    overlap: int = 200,
) -> Path:
    pdf_path = Path(pdf_path)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    chunks: list[dict] = []

    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = _clean_text(page.get_text("text"))
            if not text:
                continue

            page_chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap)
            for j, chunk in enumerate(page_chunks, start=1):
                chunks.append({
                    "page": i + 1,
                    "chunk_id": f"{i + 1}-{j}",
                    "text": chunk,
                })

        payload = {
            "source": pdf_path.name,
            "num_pages": len(doc),
            "chunk_chars": chunk_chars,
            "overlap": overlap,
            "chunking": "sentence-aware-page-bounded",
            "chunks": chunks,
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        doc.close()

    return out_json
