# Troubleshooting

## Paper Copilot cannot connect to Ollama

Open the Ollama application or run `ollama serve`, then verify the server:

```bash
curl http://localhost:11434/api/tags
```

For a server on another address, export `PAPER_COPILOT_OLLAMA_URL` before starting Streamlit.

## The model is missing

```bash
ollama pull llama3.1:8b
```

Set `PAPER_COPILOT_OLLAMA_MODEL` to use a different installed chat model.

## Semantic search cannot load its model

The first semantic or hybrid query may download `all-MiniLM-L6-v2`. Check the network connection and available disk space. For offline use, download it once, then start with `HF_HUB_OFFLINE=1`.

## An uploaded PDF has little or no searchable text

The PDF may contain scanned images without a text layer. Run OCR with a PDF tool, then upload the OCR version. Password-protected or damaged files may also fail extraction.

## Search results look stale

Select **Process / re-index library** after adding or changing papers. Older BM25-only indexes need rebuilding before semantic and hybrid search can use them.

## The app refuses to answer

The evidence guard abstains when retrieval does not sufficiently match the question, when named identifiers are absent, or when a comparison lacks evidence from two documents. Rephrase with terms used in the papers and inspect the retrieved passages.

## Installation fails

Use Python 3.11 or newer in a clean environment and install the exact dependency versions:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon, use a native arm64 Python.
