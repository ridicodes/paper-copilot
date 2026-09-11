# Week 8 — Cleanup, Documentation, and Demo Readiness

## Code health

- [ ] `python -m py_compile app.py src/*.py scripts/*.py tests/*.py`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `python scripts/smoke_test.py`
- [ ] App launches with `./run.sh`
- [ ] Ollama is reachable at the configured URL

## Retrieval regressions

- [ ] Five image-analysis stages still pass
- [ ] DP components + roles still pass
- [ ] Cross-paper method comparison uses both papers
- [ ] Unsupported YOLOv8/COCO question is refused
- [ ] Hybrid retrieval remains the default mode

## Repository hygiene

- [ ] Do not commit `outputs/`
- [ ] Do not commit research PDFs in `data/`
- [ ] Do not commit `.venv/`, caches, or `.DS_Store`
- [ ] `requirements.txt` is committed
- [ ] README reflects the current architecture
- [ ] Evaluation question set and script are committed

## Suggested final Week 8 commit

```bash
git add app.py src README.md requirements.txt .gitignore .streamlit \
  evaluation scripts tests WEEK7_EVALUATION.md WEEK8_RELEASE_CHECKLIST.md run.sh

git status
git diff --cached

git commit -m "Complete Week 7 evaluation and Week 8 project polish"
```

Push with GitHub Desktop if command-line GitHub authentication is not configured.
