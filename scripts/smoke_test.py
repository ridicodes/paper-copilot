#!/usr/bin/env python3
"""Fast Week 8 project health check (no Ollama call required)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

REQUIRED_MODULES = [
    "streamlit",
    "pymupdf",
    "rank_bm25",
    "numpy",
    "sentence_transformers",
    "requests",
]


def main() -> None:
    failures: list[str] = []

    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            print(f"[OK] dependency: {module}")
        except Exception as exc:
            failures.append(f"dependency {module}: {exc}")
            print(f"[FAIL] dependency: {module} -> {exc}")

    for path in [
        Path("app.py"),
        Path("src/index.py"),
        Path("src/ingest.py"),
        Path("src/evidence.py"),
        Path("src/citations.py"),
        Path("src/llm.py"),
        Path("src/config.py"),
        Path("evaluation/questions.json"),
    ]:
        if path.exists():
            print(f"[OK] file: {path}")
        else:
            failures.append(f"missing file: {path}")
            print(f"[FAIL] file: {path}")

    index_dir = Path("outputs/library_index")
    if index_dir.exists():
        for filename in ("bm25.pkl", "meta.json", "embeddings.npy"):
            path = index_dir / filename
            if path.exists():
                print(f"[OK] index artifact: {path}")
            else:
                failures.append(f"missing index artifact: {path}")
        meta_path = index_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(
                f"[INFO] index: {meta.get('num_documents', '?')} documents, "
                f"{meta.get('num_chunks', '?')} chunks, "
                f"model={meta.get('embedding_model', '?')}"
            )
    else:
        print("[INFO] no local library index yet; use the Streamlit UI to process PDFs.")

    if failures:
        print("\nHealth check failed:")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)

    print("\nPaper Copilot health check passed.")


if __name__ == "__main__":
    main()
