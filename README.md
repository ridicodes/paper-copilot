# Paper Copilot

Paper Copilot is a local Streamlit research-paper assistant that searches multiple PDFs with BM25, semantic embeddings, or hybrid retrieval, previews cited pages, and generates grounded answers with Ollama.

## Week 3 retrieval improvements

- sentence-aware, page-bounded chunking
- PDF hyphenation cleanup
- stopword-aware BM25 tokenization
- larger candidate pool before final ranking
- query-term coverage bonus
- exact phrase bonus
- near-duplicate suppression
- page diversity
- BM25 score + query coverage shown in the UI

## Run

```bash
source ~/.venvs/paper-copilot/bin/activate
cd ~/Documents/Github/paper-copilot
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501
```

Upload your papers and click **Process / re-index library**. Older BM25-only indexes must be re-indexed to use semantic retrieval. Select the retrieval mode under **Advanced settings**; Hybrid is the default.

## Week 6 retrieval

Semantic search uses `all-MiniLM-L6-v2`, loaded lazily from the local cache when available. The first use needs a model download. Embeddings are normalized for cosine similarity, following the [Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html). Hybrid search combines quality-adjusted BM25 and semantic ranks using reciprocal rank fusion, then applies duplicate suppression and per-document page diversity. Document/page metadata and evidence-ID citation replacement remain intact.

The answer guard accepts lexical coverage of at least 0.50 or semantic similarity of at least 0.55, excludes noisy passages, and checks explicit query identifiers against evidence. Procedural passages with similarity of at least 0.35 can supplement a strongly supported document when they describe concrete algorithm steps. Comparisons require support from at least two documents. These are prototype heuristics, not a guarantee of factual support.

Answer selection examines up to 20 candidates, prefers explicit definitions and procedures, and sends complete selected chunks to Ollama. Evidence cards continue to show short snippets. Citation validation checks both allowed evidence IDs and sentence completeness; invalid answers use source excerpts with trusted page citations. A targeted private-SGD check also falls back when the generated answer omits key algorithm operations. Citation validity does not establish factual entailment.

### Validation (2026-09-10)

The handoff's two-paper library contains 114 chunks with 384-dimensional embeddings. All seven handoff questions plus one privacy paraphrase were exercised against local `llama3.1:8b`:

- Image-analysis stages: all five stages, CV page 2.
- Region-based segmentation: both weaknesses, CV page 5.
- Pattern recognition: definition and role, CV page 3.
- Computer-vision applications: cited applications from the CV paper.
- YOLOv8/COCO: blocked before model invocation; rejection also tested in all three retrieval modes.
- Private training and its paraphrase: complete page 3 algorithm supplied; incomplete generated explanations replaced with cited excerpts covering gradients, clipping, averaging, noise, updates, and privacy accounting.
- Cross-paper comparison: cited excerpts contrast private neural-network training with vision applications when generated prose fails citation validation.

The final recorded live responses used the extractive fallback for private training, its paraphrase, and the comparison. Automated checks cover citation failures, method-prompt completeness, retrieval/index compatibility, mode changes, answer generation, and PDF page viewing for both documents. Thirteen tests pass with the local-library checks enabled.

Some relevant paraphrases remain below the conservative threshold, and Semantic-only comparison retrieval can abstain. Broader evaluation and threshold calibration remain Week 7 work. This small benchmark does not establish that Hybrid is universally better than BM25.

Run deterministic tests without a model download:

```bash
python -m unittest discover -s tests -v
```

Also run local-library retrieval and Streamlit checks when the handoff's PDFs, index, and cached model are present:

```bash
PAPER_COPILOT_LIBRARY_TESTS=1 HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
```

Repeat the live benchmark with Ollama running (run from the repository root):

```bash
HF_HUB_OFFLINE=1 python scripts/validate_week6.py --live
```

The report defaults to `/tmp/paper-copilot-week6-validation.json`; generated reports and indexes should not be committed.
