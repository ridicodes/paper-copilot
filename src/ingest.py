import json
from pathlib import Path
import fitz  
def ingest_pdf(pdf_path: str | Path, out_json: str | Path):
    pdf_path = Path(pdf_path)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    chunks = []
    chunk_id = 0
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        if not text:
            continue
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        buff = ""
        for p in paras:
            p = " ".join(p.replace("\n", " ").split()).strip()
            if len(buff) + len(p) + 1 <= 900:
                buff = (buff + " " + p).strip()
            else:
                chunks.append({"chunk_id": chunk_id, "page": i + 1, "text": buff})
                chunk_id += 1
                buff = p
        if buff:
            chunks.append({"chunk_id": chunk_id, "page": i + 1, "text": buff})
            chunk_id += 1

    payload = {"source_pdf": pdf_path.name, "num_pages": len(doc), "chunks": chunks}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_json
