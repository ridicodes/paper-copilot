# Week 7 — Evaluation and Regression Testing

Week 7 turns Paper Copilot from a demo into a system whose retrieval behavior can be measured and repeatedly checked.

## What is evaluated

The benchmark in `evaluation/questions.json` contains:

- direct single-paper factual/list retrieval;
- conceptual single-paper retrieval;
- differential-privacy methodology retrieval;
- cross-paper semantic comparisons;
- cross-paper method/purpose questions;
- one deliberately unsupported YOLOv8/COCO question.

The retrieval evaluator compares **BM25**, **semantic**, and **hybrid** search with the same question set.

## Metrics

- **Document recall** — whether the expected paper(s) appear in the top-k evidence.
- **Page hit** — whether a known supporting page appears when a page-level gold label exists.
- **MRR** — how early the first known relevant result appears.
- **Safe refusal** — for the unsupported negative control, whether Paper Copilot's existing evidence gate refuses support.

These are retrieval/safety regression metrics, not claims that every possible answer is objectively graded.

## Run the benchmark

First process/re-index the two test papers in the app, then:

```bash
python scripts/evaluate_retrieval.py --k 5
```

The CSV is written to:

```text
evaluation/results/retrieval_metrics.csv
```

You can test only hybrid retrieval with:

```bash
python scripts/evaluate_retrieval.py --methods hybrid --k 5
```

## Manual answer-quality regression set

After retrieval evaluation, verify these in the UI:

1. `What are the main stages of image analysis?`
   - Expect all five stages and the computer-vision page-2 citation.
2. `What are the main components of the differentially private deep learning approach, and what role does each component play?`
   - Expect DP-SGD, moments accountant, and hyperparameter tuning with grounded roles.
3. `Compare the main techniques used in the two papers. How does each technique contribute to solving the paper's central problem?`
   - Expect evidence from both papers and no one-paper monopoly.
4. `What accuracy does the YOLOv8 model achieve on the COCO dataset?`
   - Expect a refusal/insufficient-evidence outcome, not an invented number.

Do not tune a threshold only to make one question pass. A change is accepted only if the regression set remains safe.
