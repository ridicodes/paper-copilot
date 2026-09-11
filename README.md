# Paper Copilot

Paper Copilot is a local Streamlit assistant for searching and comparing research papers. It indexes PDF text, retrieves evidence with BM25, semantic, or hybrid search, and asks a local Ollama model to answer with page citations. Answers are rejected or replaced with cited excerpts when the available evidence or model citations fail validation.

## Features

- Multi-PDF research library with document and page metadata
- BM25, semantic, and reciprocal-rank-fusion hybrid retrieval
- Evidence sufficiency checks for unsupported and cross-paper questions
- Local answer generation through Ollama
- Citation validation and extractive fallback
- Cited PDF page previews
- A labelled 32-question retrieval and answer benchmark

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com/) for generated answers
- About 500 MB of free space for Python packages and the embedding model

## Setup

```bash
git clone <repository-url>
cd paper-copilot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull llama3.1:8b
```

Start Ollama, then launch the app:

```bash
ollama serve
python -m streamlit run app.py
```

Open the displayed local URL, upload one or more PDFs, and select **Process / re-index library**. The first semantic or hybrid search downloads `all-MiniLM-L6-v2` unless it is already cached.

Runtime settings can be overridden with the environment variables listed in [`.env.example`](.env.example). Export them in the shell before starting Streamlit.

## Validation

```bash
python -m unittest discover -s tests -v
```

With the two-paper development library and embedding model present, run the integration suite offline:

```bash
PAPER_COPILOT_LIBRARY_TESTS=1 HF_HUB_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

The Week 7 benchmark contains 32 labelled questions. Hybrid retrieval achieved 100% Hit@5, unsupported-question rejection, comparison document coverage, and final grounded-answer rate on this small local dataset. BM25 had the best Hit@1 and MRR. See [WEEK7_EVALUATION.md](WEEK7_EVALUATION.md) for the protocol, results, and limitations.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Demo guide](docs/DEMO.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

Uploaded papers, extracted text, indexes, and model traffic remain on the local machine. Paper Copilot is a prototype research aid; verify cited pages before relying on an answer.
