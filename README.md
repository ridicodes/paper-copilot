# Paper Copilot

Paper Copilot is a local Streamlit research-paper assistant that uploads a PDF, retrieves evidence with BM25, previews cited pages, and generates grounded answers with Ollama.

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

After replacing the Week 3 files, upload the PDF again and click **Process / re-index PDF** so the new chunking and tokenizer are used.
