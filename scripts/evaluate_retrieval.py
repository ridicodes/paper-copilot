#!/usr/bin/env python3
"""Week 7 retrieval benchmark for Paper Copilot.

Runs benchmark questions through BM25, semantic, and hybrid retrieval.
Reports document recall, page hit-rate, reciprocal rank, and whether
unsupported questions are safely rejected.

This script does NOT call Ollama.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evidence import evidence_is_sufficient
from src.index import (
    hybrid_search,
    is_comparison_question,
    search,
    semantic_search,
)


# ============================================================
# RETRIEVAL METHODS
# ============================================================

METHODS = {
    "bm25": search,
    "semantic": semantic_search,
    "hybrid": hybrid_search,
}


# ============================================================
# LOAD EVALUATION QUESTIONS
# ============================================================

def load_cases(path: Path) -> list[dict]:
    """Load evaluation cases from JSON."""

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, list):
        raise ValueError(
            "Evaluation file must contain a JSON list."
        )

    return data


# ============================================================
# GOLD RESULT CHECK
# ============================================================

def result_is_gold(
    result: dict,
    case: dict,
) -> bool:
    """Return True when a retrieved result matches expected doc/page."""

    expected_docs = set(
        case.get(
            "expected_documents",
            [],
        )
    )

    document = str(
        result.get(
            "document",
            "",
        )
    )

    if document not in expected_docs:
        return False

    pages = (
        case.get(
            "expected_pages",
            {},
        )
        .get(
            document,
            [],
        )
    )

    # If no specific page is required, correct document is enough.
    if not pages:
        return True

    try:
        result_page = int(
            result.get(
                "page",
                -1,
            )
        )
    except (TypeError, ValueError):
        return False

    accepted_pages = {
        int(page)
        for page in pages
    }

    return result_page in accepted_pages


# ============================================================
# CASE SCORING
# ============================================================

def score_case(
    case: dict,
    results: list[dict],
) -> dict:
    """Calculate retrieval metrics for one evaluation question."""

    question = str(
        case.get(
            "question",
            "",
        )
    )

    # --------------------------------------------------------
    # Unsupported / negative-control question
    # --------------------------------------------------------

    if case.get(
        "expected_no_support",
        False,
    ):

        sufficient = evidence_is_sufficient(
            question,
            results,
            comparison=is_comparison_question(
                question
            ),
        )

        return {
            "document_recall": 1.0,
            "page_hit": 1.0,
            "mrr": 1.0,
            "safe_refusal": (
                0.0
                if sufficient
                else 1.0
            ),
        }

    expected_docs = set(
        case.get(
            "expected_documents",
            [],
        )
    )

    retrieved_docs = {
        str(
            result.get(
                "document",
                "",
            )
        )
        for result in results
    }

    # --------------------------------------------------------
    # Document recall
    # --------------------------------------------------------

    if expected_docs:
        document_recall = (
            len(
                expected_docs
                & retrieved_docs
            )
            / len(
                expected_docs
            )
        )
    else:
        document_recall = 1.0

    # --------------------------------------------------------
    # Page hit rate
    # --------------------------------------------------------

    expected_pages = case.get(
        "expected_pages",
        {},
    )

    page_checks: list[float] = []

    for document, pages in expected_pages.items():

        if not pages:
            continue

        accepted_pages = {
            int(page)
            for page in pages
        }

        hit = any(
            str(
                result.get(
                    "document",
                    "",
                )
            )
            == document
            and int(
                result.get(
                    "page",
                    -1,
                )
            )
            in accepted_pages
            for result in results
        )

        page_checks.append(
            1.0
            if hit
            else 0.0
        )

    if page_checks:
        page_hit = mean(
            page_checks
        )
    else:
        page_hit = document_recall

    # --------------------------------------------------------
    # Mean Reciprocal Rank contribution
    # --------------------------------------------------------

    reciprocal_rank = 0.0

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if result_is_gold(
            result,
            case,
        ):
            reciprocal_rank = (
                1.0
                / float(rank)
            )
            break

    return {
        "document_recall": document_recall,
        "page_hit": page_hit,
        "mrr": reciprocal_rank,
        "safe_refusal": 1.0,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_result_preview(
    result: dict,
) -> str:
    """Create a short readable preview for debugging."""

    document = str(
        result.get(
            "document",
            "Unknown paper",
        )
    )

    page = result.get(
        "page",
        "?"
    )

    text = str(
        result.get(
            "text",
            "",
        )
    )

    text = " ".join(
        text.split()
    )

    if len(text) > 120:
        text = (
            text[:117]
            + "..."
        )

    return (
        f"{document} p.{page}: "
        f"{text}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Paper Copilot retrieval using "
            "BM25, semantic, and hybrid search."
        )
    )

    parser.add_argument(
        "--index",
        default="outputs/library_index",
        help="Path to the combined Paper Copilot index.",
    )

    parser.add_argument(
        "--questions",
        default="evaluation/questions.json",
        help="Evaluation question JSON file.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved passages per question.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(
            METHODS.keys()
        ),
        default=[
            "bm25",
            "semantic",
            "hybrid",
        ],
        help="Retrieval methods to evaluate.",
    )

    parser.add_argument(
        "--output",
        default=(
            "evaluation/results/"
            "retrieval_metrics.csv"
        ),
        help="CSV output path.",
    )

    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Print retrieved passages for each question.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve paths relative to project root
    # --------------------------------------------------------

    index_dir = (
        PROJECT_ROOT
        / args.index
    ).resolve()

    questions_path = (
        PROJECT_ROOT
        / args.questions
    ).resolve()

    output_path = (
        PROJECT_ROOT
        / args.output
    ).resolve()

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    if not (
        index_dir
        / "meta.json"
    ).exists():

        raise SystemExit(
            "\nERROR: Paper Copilot index not found.\n"
            f"Expected: {index_dir}\n\n"
            "Open Paper Copilot and process/re-index "
            "the research library first."
        )

    if not (
        index_dir
        / "bm25.pkl"
    ).exists():

        raise SystemExit(
            "\nERROR: bm25.pkl is missing from the index.\n"
            f"Expected: {index_dir / 'bm25.pkl'}"
        )

    if not (
        index_dir
        / "embeddings.npy"
    ).exists():

        raise SystemExit(
            "\nERROR: embeddings.npy is missing from the index.\n"
            f"Expected: {index_dir / 'embeddings.npy'}"
        )

    if not questions_path.exists():

        raise SystemExit(
            "\nERROR: Evaluation questions file not found.\n"
            f"Expected: {questions_path}"
        )

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    cases = load_cases(
        questions_path
    )

    if not cases:
        raise SystemExit(
            "No evaluation cases were found."
        )

    print(
        f"\nLoaded {len(cases)} evaluation cases."
    )

    print(
        f"Index: {index_dir}"
    )

    print(
        f"Top-k: {args.k}"
    )

    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    rows: list[dict] = []

    for method_name in args.methods:

        method = METHODS[
            method_name
        ]

        print(
            "\n"
            + "=" * 78
        )

        print(
            method_name.upper()
        )

        print(
            "=" * 78
        )

        for case in cases:

            case_id = str(
                case.get(
                    "id",
                    "unknown",
                )
            )

            question = str(
                case.get(
                    "question",
                    "",
                )
            )

            print(
                f"\n[{case_id}]"
            )

            print(
                question
            )

            try:

                results = method(
                    index_dir,
                    question,
                    k=args.k,
                )

            except Exception as exc:

                print(
                    f"  ERROR during retrieval: {exc}"
                )

                rows.append(
                    {
                        "method": method_name,
                        "case_id": case_id,
                        "question": question,
                        "document_recall": 0.0,
                        "page_hit": 0.0,
                        "mrr": 0.0,
                        "safe_refusal": 0.0,
                        "error": str(
                            exc
                        ),
                    }
                )

                continue

            scores = score_case(
                case,
                results,
            )

            row = {
                "method": method_name,
                "case_id": case_id,
                "question": question,
                "document_recall": scores[
                    "document_recall"
                ],
                "page_hit": scores[
                    "page_hit"
                ],
                "mrr": scores[
                    "mrr"
                ],
                "safe_refusal": scores[
                    "safe_refusal"
                ],
                "error": "",
            }

            rows.append(
                row
            )

            print(
                "  "
                f"document_recall="
                f"{scores['document_recall']:.2f} | "
                f"page_hit="
                f"{scores['page_hit']:.2f} | "
                f"MRR="
                f"{scores['mrr']:.2f} | "
                f"safe_refusal="
                f"{scores['safe_refusal']:.2f}"
            )

            if args.show_results:

                print(
                    "  Retrieved:"
                )

                for rank, result in enumerate(
                    results,
                    start=1,
                ):

                    print(
                        f"    {rank}. "
                        f"{format_result_preview(result)}"
                    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "method",
        "case_id",
        "question",
        "document_recall",
        "page_hit",
        "mrr",
        "safe_refusal",
        "error",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 78
    )

    for method_name in args.methods:

        group = [
            row
            for row in rows
            if row[
                "method"
            ]
            == method_name
            and not row[
                "error"
            ]
        ]

        if not group:

            print(
                f"{method_name:<10} "
                "No successful evaluation cases."
            )

            continue

        avg_document_recall = mean(
            float(
                row[
                    "document_recall"
                ]
            )
            for row in group
        )

        avg_page_hit = mean(
            float(
                row[
                    "page_hit"
                ]
            )
            for row in group
        )

        avg_mrr = mean(
            float(
                row[
                    "mrr"
                ]
            )
            for row in group
        )

        avg_safe_refusal = mean(
            float(
                row[
                    "safe_refusal"
                ]
            )
            for row in group
        )

        print(
            f"{method_name:<10} "
            f"doc_recall={avg_document_recall:.3f} | "
            f"page_hit={avg_page_hit:.3f} | "
            f"MRR={avg_mrr:.3f} | "
            f"safe_refusal={avg_safe_refusal:.3f}"
        )

    print(
        "\nEvaluation complete."
    )

    print(
        f"Saved CSV: {output_path}"
    )


if __name__ == "__main__":
    main()