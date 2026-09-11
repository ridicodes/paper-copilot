# Week 7 evaluation

## Scope

This evaluation uses the two-paper, 114-chunk handoff library and 32 manually labelled questions in `evaluation/week7_questions.json`: 26 supported questions (facts, definitions, lists, limitations, methodology, paraphrases, and comparisons) and six unsupported questions. A result is relevant when its document and page match an accepted label. Some questions allow multiple pages because the source repeats or continues the same evidence.

The evaluator retrieves 20 results from BM25, Semantic, and Hybrid. It reports Hit@k, reciprocal rank, whether both labelled documents appear for comparisons, evidence-gate decisions, and the proportion of top-five results whose noise penalty marks them as reference-like. `evaluation/week7_results.json` is generated locally and ignored by Git.

## Retrieval results

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@20 | Supported accepted | Unsupported rejected | Both papers at 5 | Noise in top 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 88.5% | 96.2% | 96.2% | 0.923 | 96.2% | 100% | 100% | 3.1% |
| Semantic | 57.7% | 88.5% | 92.3% | 0.738 | 100% | 100% | 50% | 0% |
| Hybrid | 76.9% | 92.3% | 100% | 0.859 | 100% | 100% | 100% | 0% |

BM25 ranks exact matches best on this small corpus. Hybrid has the best retrieval breadth: every supported question has a labelled page in the top five, both comparisons contain both papers, all unsupported questions are rejected, and no top-five result is marked as reference noise. Semantic alone is weaker for exact definitions and one comparison. These measurements support Hybrid as the application default while retaining BM25 as a useful mode; they do not show that Hybrid dominates every ranking metric.

## Tuning decisions

The labelled dataset was frozen before tuning. The final pass made three targeted changes:

- Comparison queries lead with each document's strongest fused candidate before resuming normal fused order. This prevents a single paper from monopolizing the evidence window.
- Methodology boosting no longer treats every `how can` question or comparison as procedural. This removed unrelated private-training passages from general visual questions.
- Currency words are treated as explicit query anchors. This fixed the false acceptance of the unsupported training-cost-in-dollars question.

The existing thresholds remain 0.50 lexical similarity, 0.55 semantic similarity, and 0.35 semantic similarity for supplementary methodology evidence. Comparison sufficiency accepts clean passages at 0.20 lexical coverage or 0.30 semantic similarity, and still requires two documents. The data did not justify changing the core thresholds.

## Answer evaluation

`scripts/evaluate_answers.py` runs Hybrid retrieval and local Ollama generation over all 32 questions. It measures citation validity and completeness, concept coverage from `evaluation/week7_answer_rubric.json`, citation precision against labelled pages, fallback use, and unsupported abstention. The generated `evaluation/week7_answer_results.json` is ignored by Git.

Run it with Ollama and `llama3.1:8b` available:

```bash
source ~/.venvs/paper-copilot/bin/activate
HF_HUB_OFFLINE=1 python scripts/evaluate_answers.py
```

## Live answer results

All 32 questions were run with local `llama3.1:8b`. The six unsupported questions were rejected before generation. Results for the 26 supported questions were:

| Metric | Result |
|---|---:|
| Raw Ollama citation-valid rate | 80.8% |
| Final concept coverage | 100% |
| Final citation-to-labelled-page precision | 100% |
| Citation-safe fallback rate | 26.9% |
| Unsupported abstention rate | 100% |
| Final grounded-answer rate | 100% |

“Raw citation-valid” measures whether Ollama followed evidence-ID and sentence-level citation rules without intervention. “Final” metrics include the deterministic cited-excerpt fallback, which activates for invalid citations or an incomplete private-training procedure. Concept coverage is a lexical rubric over expected concepts; it is reproducible but does not prove semantic correctness. Citation precision confirms that cited evidence comes from manually accepted pages; it does not independently prove that every claim is entailed by its citation.

## Reproduction

```bash
source ~/.venvs/paper-copilot/bin/activate
HF_HUB_OFFLINE=1 python scripts/evaluate_retrieval.py
HF_HUB_OFFLINE=1 python scripts/evaluate_answers.py
PAPER_COPILOT_LIBRARY_TESTS=1 HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
```

The evaluation is intentionally limited to two papers and 32 questions authored from those papers. The same labelled set was used to identify and verify targeted fixes, so it is not a held-out test set. The answer rubric checks required concepts by phrase alternatives and cannot judge every nuance of correctness. Week 8 documentation should present these as local regression-benchmark results, not general retrieval or answer-quality guarantees.
