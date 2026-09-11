"""Runtime configuration shared by the Streamlit app and service modules."""

import os
from pathlib import Path


DEFAULT_OLLAMA_MODEL = os.getenv("PAPER_COPILOT_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("PAPER_COPILOT_OLLAMA_URL", "http://localhost:11434").rstrip("/")
LIBRARY_INDEX_DIR = Path(os.getenv("PAPER_COPILOT_INDEX_DIR", "outputs/library_index"))

MAX_ANSWER_PASSAGES = 4
COMPARISON_SEARCH_K = 20

BAD_PATTERNS = (
    "issn",
    "international journal",
    "copyright",
    "all rights reserved",
    "no researchers usefulness definition",
)
