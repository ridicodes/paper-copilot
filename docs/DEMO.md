# Demo guide

## Before the demo

1. Start Ollama and confirm `llama3.1:8b` is installed with `ollama list`.
2. Activate the virtual environment and run `python -m streamlit run app.py`.
3. Upload at least two text-based research PDFs.
4. Select **Process / re-index library** and wait for indexing to finish.
5. Leave retrieval mode on **Hybrid**.

## Suggested flow

1. Ask a direct fact or definition question. Point out that each evidence card names the paper and page.
2. Open a cited-page preview to verify the answer against the PDF.
3. Ask a method question beginning with “How…”. Explain that the app selects procedure evidence and checks that the result covers the retrieved steps.
4. Ask “How do the two papers approach their problems differently?” Show that comparison evidence comes from both papers.
5. Ask for a detail absent from the papers. The app should abstain before contacting Ollama.
6. Switch between BM25, Semantic, and Hybrid in advanced settings and repeat a query to show retrieval differences.

## What to explain

- Papers and prompts stay local: the answer model is served by Ollama.
- Search results are evidence candidates; page citations let the user verify them.
- Citation checks catch malformed or invented evidence IDs. They do not prove every sentence is entailed, so the source page remains the final reference.
- The benchmark is deliberately small and measures this development library, not every academic domain.
