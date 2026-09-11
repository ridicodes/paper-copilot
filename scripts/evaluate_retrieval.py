"""Measure Paper Copilot retrieval against the labelled Week 7 set."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evidence import evidence_is_sufficient
from src.evidence import (LEXICAL_THRESHOLD, SEMANTIC_THRESHOLD,
                          METHODOLOGY_SEMANTIC_THRESHOLD)
from src.index import hybrid_search, search, semantic_search

METHODS = {"bm25": search, "semantic": semantic_search, "hybrid": hybrid_search}


def relevant(result: dict, labels: list[dict]) -> bool:
    return any(result.get("document") == label["document"]
               and int(result.get("page", 0)) in label["pages"] for label in labels)


def required_documents_found(results: list[dict], labels: list[dict], k: int) -> bool:
    found = {r.get("document") for r in results[:k] if relevant(r, labels)}
    return all(label["document"] in found for label in labels)


def evaluate_case(case: dict, results: list[dict]) -> dict:
    ranks = [i for i, result in enumerate(results, 1) if relevant(result, case["relevant"])]
    comparison = case["type"] == "comparison"
    sufficient = evidence_is_sufficient(case["query"], results, comparison=comparison)
    noise = [r for r in results[:5] if float(r.get("noise_penalty", 0)) >= 3.0]
    return {
        "first_relevant_rank": min(ranks) if ranks else None,
        "hit_at_1": bool(ranks and min(ranks) <= 1),
        "hit_at_3": bool(ranks and min(ranks) <= 3),
        "hit_at_5": bool(ranks and min(ranks) <= 5),
        "all_documents_at_5": required_documents_found(results, case["relevant"], 5),
        "sufficient": sufficient,
        "correct_rejection": not sufficient if not case["supported"] else None,
        "noise_top_5": len(noise),
        "top_5": [{"rank": r["rank"], "document": r["document"], "page": r["page"],
                   "coverage": round(float(r.get("coverage", 0)), 4),
                   "semantic_score": round(float(r.get("semantic_score", 0)), 4),
                   "noise_penalty": float(r.get("noise_penalty", 0))} for r in results[:5]],
    }


def aggregate(rows: list[dict]) -> dict:
    supported = [r for r in rows if r["supported"]]
    unsupported = [r for r in rows if not r["supported"]]
    comparisons = [r for r in supported if r["type"] == "comparison"]
    metric = lambda key, items: (round(sum(bool(r[key]) for r in items) / len(items), 4)
                                 if items else None)
    return {
        "supported_questions": len(supported),
        "unsupported_questions": len(unsupported),
        "hit_at_1": metric("hit_at_1", supported),
        "hit_at_3": metric("hit_at_3", supported),
        "hit_at_5": metric("hit_at_5", supported),
        "mrr_at_20": round(sum(1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0
                               for r in supported) / len(supported), 4),
        "supported_acceptance": metric("sufficient", supported),
        "unsupported_rejection": metric("correct_rejection", unsupported),
        "comparison_all_documents_at_5": metric("all_documents_at_5", comparisons),
        "noise_rate_top_5": round(sum(r["noise_top_5"] for r in rows) / (5 * len(rows)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/week7_questions.json")
    parser.add_argument("--index", type=Path, default=ROOT / "outputs/library_index")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/week7_results.json")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    cases = json.loads(args.dataset.read_text())
    report = {"dataset": str(args.dataset.relative_to(ROOT)), "index": str(args.index),
              "k": args.k, "thresholds": {
                  "lexical": LEXICAL_THRESHOLD,
                  "semantic": SEMANTIC_THRESHOLD,
                  "methodology_semantic": METHODOLOGY_SEMANTIC_THRESHOLD,
                  "comparison_coverage": 0.20,
                  "comparison_semantic": 0.30,
              }, "methods": {}}
    for name, method in METHODS.items():
        rows = []
        for case in cases:
            result = evaluate_case(case, method(args.index, case["query"], k=args.k))
            rows.append({"id": case["id"], "type": case["type"], "query": case["query"],
                         "supported": case["supported"], **result})
        report["methods"][name] = {"summary": aggregate(rows), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({name: value["summary"] for name, value in report["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
