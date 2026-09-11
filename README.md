# Paper Copilot

Paper Copilot is a local-first research-paper assistant that searches across multiple PDFs, shows page-level evidence, and generates answers that are constrained to retrieved evidence.

The project combines **BM25 lexical retrieval**, **sentence-transformer semantic retrieval**, and **hybrid Reciprocal Rank Fusion (RRF)**. Answers are generated locally through **Ollama**, then checked for citation completeness and basic claim-to-evidence grounding before being shown.

## Current capabilities

- Upload and index multiple research papers.
- Sentence-aware, page-bounded PDF chunking.
- BM25, semantic, and hybrid retrieval modes.
- `all-MiniLM-L6-v2` embeddings stored locally.
- Cross-paper comparison detection and document-aware evidence selection.
- Method/technique + purpose evidence aggregation across multiple passages.
- Evidence cards with paper name, page number, lexical coverage, retrieval diagnostics, and PDF page preview.
- Local Ollama generation using `llama3.1:8b` by default.
- Evidence-ID citations (`[E1]`, `[E2]`, …) mapped back to trusted paper/page metadata in Python.
- Citation completeness and lightweight claim-to-evidence grounding checks.
- Conservative citation-safe fallbacks when generated text fails validation.
- Unsupported-question safeguard.
- Week 7 retrieval benchmark and regression tests.

## Architecture

```text
PDFs
  ↓
PyMuPDF page extraction
  ↓
Sentence-aware, page-bounded chunks
  ↓
┌─────────────────────┬────────────────────────┐
│ BM25 lexical index  │ MiniLM embeddings      │
└──────────┬──────────┴───────────┬────────────┘
           └──── Reciprocal Rank Fusion ───────┘
                          ↓
                Evidence selection
                          ↓
                 Sufficiency checks
                          ↓
                Ollama / safe fallback
                          ↓
             Citation + grounding checks
                          ↓
             Answer + page-level evidence
```

## Project structure

```text
paper-copilot/
├── app.py
├── src/
│   ├── citations.py
│   ├── config.py
│   ├── evidence.py
│   ├── index.py
│   ├── ingest.py
│   └── llm.py
├── evaluation/
│   └── questions.json
├── scripts/
│   ├── evaluate_retrieval.py
│   └── smoke_test.py
├── tests/
│   └── test_core.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── run.sh
├── WEEK7_EVALUATION.md
└── WEEK8_RELEASE_CHECKLIST.md
```

`data/` and `outputs/` are local runtime directories and are ignored by Git.

## Setup

### 1. Create or activate a Python environment

For the development machine used for this project:

```bash
cd ~/Documents/Github/paper-copilot
source ~/.venvs/paper-copilot/bin/activate
```

For a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install and run Ollama

Pull the default model once:

```bash
ollama pull llama3.1:8b
```

Confirm Ollama is available:

```bash
curl http://localhost:11434/api/tags
```

If Ollama is not already running, open the Ollama app or run:

```bash
ollama serve
```

### 3. Start Paper Copilot

```bash
./run.sh
```

or:

```bash
python -m streamlit run app.py --server.port 8501
```

Then open the local Streamlit URL shown in Terminal.

## Using the app

1. Upload one or more PDFs.
2. Click **Process / re-index library**.
3. Ask a question.
4. Keep **Hybrid** selected for the default retrieval experience.
5. Click **Search evidence** to inspect retrieval before generation.
6. Click **Generate answer**.
7. Expand retrieval details or use **View page** to verify the original source.

## Retrieval modes

### BM25

Strong when the wording of the question overlaps closely with the paper. It remains an important lexical baseline.

### Semantic

Uses `all-MiniLM-L6-v2` embeddings and cosine similarity to retrieve conceptually related text even when wording differs.

### Hybrid

The default mode. Paper Copilot fuses BM25 and semantic rankings with Reciprocal Rank Fusion rather than directly adding incompatible raw score scales. Noise penalties and diversity logic reduce bibliography/reference dominance and duplicate passages.

## Grounding and safety design

Paper Copilot deliberately separates retrieval relevance from answer support.

A passage can be semantically similar without being sufficient evidence. Before generation, the project applies evidence sufficiency checks. Cross-paper questions require support from multiple documents, and method/purpose comparisons may aggregate complementary passages within the same paper.

The LLM receives evidence IDs rather than trusted citations. After generation, Python validates the IDs and converts them to paper/page citations. Citation completeness and lightweight lexical grounding checks catch common cases where a valid citation is attached to an unsupported claim. If generation fails validation, Paper Copilot can produce a conservative citation-safe fallback instead.

This is a research/portfolio prototype, not a formal factual-verification system. Users should inspect the cited page when correctness matters.

## Week 7 evaluation

The benchmark compares BM25, semantic, and hybrid retrieval on known single-paper, cross-paper, methodology, and unsupported queries.

```bash
python scripts/evaluate_retrieval.py --k 5
```

See [`WEEK7_EVALUATION.md`](WEEK7_EVALUATION.md) for the metrics and regression set.

## Development checks

Compile everything:

```bash
python -m py_compile app.py src/*.py scripts/*.py tests/*.py
```

Run unit regressions:

```bash
python -m unittest discover -s tests -v
```

Run the local environment health check:

```bash
python scripts/smoke_test.py
```

## Environment variables

Optional overrides:

```bash
export PAPER_COPILOT_OLLAMA_MODEL="llama3.1:8b"
export PAPER_COPILOT_OLLAMA_URL="http://localhost:11434"
export PAPER_COPILOT_INDEX_DIR="outputs/library_index"
export PAPER_COPILOT_PORT="8501"
```

## Known limitations

- The grounding validator is intentionally lightweight; it is not a full natural-language-inference model.
- Retrieval quality still depends on PDF text extraction quality and chunk boundaries.
- Tables, equations, and heavily scanned PDFs are not handled as richly as normal selectable text.
- The semantic model is downloaded/cached locally the first time it is needed.
- The answer generator is limited by the local Ollama model and may trigger the safer extractive fallback.
- Evidence shown in the UI is the user-selected top-k, while answer generation can use a deeper internal candidate pool.

## Project status

- Weeks 1–5: ingestion, evidence UI, retrieval improvements, citations, multi-PDF library.
- Week 6: hybrid BM25 + semantic retrieval and comparison-aware evidence handling.
- Week 7: benchmark/evaluation and regression testing.
- Week 8: cleanup, documentation, reproducible setup, health checks, and demo readiness.

The project is now at the end of the planned Week 8 milestone. Week 9 can focus on portfolio polish, richer evaluation, deployment options, and stretch features rather than core retrieval rewrites.
