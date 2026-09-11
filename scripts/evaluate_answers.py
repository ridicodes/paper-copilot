"""Evaluate grounded Hybrid answers with the local Ollama model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.index import hybrid_search


def labelled(item: dict, labels: list[dict]) -> bool:
    return any(item.get("document") == label["document"]
               and int(item.get("page", 0)) in label["pages"] for label in labels)


def concept_coverage(answer: str, groups: list[list[str]]) -> float:
    low = answer.lower()
    return round(sum(any(term.lower() in low for term in group) for group in groups)
                 / len(groups), 4) if groups else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/week7_questions.json")
    parser.add_argument("--rubric", type=Path, default=ROOT / "evaluation/week7_answer_rubric.json")
    parser.add_argument("--index", type=Path, default=ROOT / "outputs/library_index")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/week7_answer_results.json")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ids", help="Comma-separated case IDs to rerun and merge into output")
    parser.add_argument("--rescore", action="store_true",
                        help="Recompute rubric coverage from an existing result without Ollama")
    args = parser.parse_args()

    namespace = {}
    source = (ROOT / "app.py").read_text().split("st.set_page_config(")[0]
    exec(compile(source, str(ROOT / "app.py"), "exec"), namespace)
    cases = json.loads(args.dataset.read_text())
    rubrics = json.loads(args.rubric.read_text())
    if args.rescore:
        report = json.loads(args.output.read_text())
        rows = report["cases"]
        for row in rows:
            if row["supported"] and row.get("final_answer"):
                row["concept_coverage"] = concept_coverage(
                    row["final_answer"], rubrics.get(row["id"], []))
        supported = [row for row in rows if row["supported"]]
        unsupported = [row for row in rows if not row["supported"]]
        mean = lambda key, data: round(sum(float(row.get(key, 0)) for row in data) / len(data), 4)
        report["summary"] = {
            "supported_answers": len(supported), "unsupported_questions": len(unsupported),
            "citation_valid_rate": mean("citation_valid", supported),
            "mean_concept_coverage": mean("concept_coverage", supported),
            "mean_citation_label_precision": mean("citation_label_precision", supported),
            "fallback_rate": mean("fallback_used", supported),
            "unsupported_abstention_rate": mean("correct_abstention", unsupported),
            "final_grounded_rate": round(sum(bool(row.get("final_answer"))
                and row.get("concept_coverage") == 1.0
                and row.get("citation_label_precision") == 1.0 for row in supported)
                / len(supported), 4),
        }
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report["summary"], indent=2))
        return
    previous = {}
    if args.ids and args.output.exists():
        previous = {row["id"]: row for row in json.loads(args.output.read_text()).get("cases", [])}
    selected_ids = set(args.ids.split(",")) if args.ids else None
    rows = []
    for case in cases:
        if selected_ids is not None and case["id"] not in selected_ids:
            if case["id"] not in previous:
                raise ValueError(f"No prior result exists for {case['id']}; run the full evaluation first")
            rows.append(previous[case["id"]])
            continue
        results = hybrid_search(args.index, case["query"], k=namespace["COMPARISON_SEARCH_K"])
        sufficient = namespace["evidence_is_sufficient"](results, case["query"])
        row = {"id": case["id"], "type": case["type"], "supported": case["supported"],
               "sufficient": sufficient}
        if not sufficient:
            row.update(abstained=True, correct_abstention=not case["supported"])
        else:
            selected = namespace["select_answer_evidence"](case["query"], results)
            prompt, evidence = namespace["build_answer_prompt"](case["query"], selected)
            raw = namespace["normalize_answer"](
                namespace["ollama_chat"](prompt, model=args.model), case["query"])
            if raw.startswith(("Could not connect", "Ollama timed out", "Ollama error")):
                raise RuntimeError(raw)
            citation_valid = namespace["evidence_ids_are_valid"](raw, evidence)
            method_complete = (not namespace["is_methodology_question"](case["query"])
                or namespace["is_cross_document_question"](case["query"])
                or namespace["methodology_answer_is_complete"](raw, selected))
            answer_valid = citation_valid and method_complete
            fallback = not answer_valid and raw != "Not found in the provided evidence."
            final = (namespace["replace_evidence_ids_with_citations"](raw, evidence)
                     if answer_valid else raw if raw == "Not found in the provided evidence."
                     else namespace["make_extractive_answer"](case["query"], selected))
            used = namespace["extract_evidence_ids"](raw)
            cited = [evidence[item] for item in used if item in evidence]
            assessed = selected if fallback else cited
            row.update(abstained=raw == "Not found in the provided evidence.",
                       correct_abstention=False, raw_answer=raw, final_answer=final,
                       citation_valid=citation_valid, methodology_complete=method_complete,
                       fallback_used=fallback,
                       citation_label_precision=round(sum(labelled(x, case["relevant"])
                           for x in assessed) / len(assessed), 4) if assessed else 0.0,
                       concept_coverage=concept_coverage(final, rubrics.get(case["id"], [])))
        rows.append(row)
        partial = rows + [previous[c["id"]] for c in cases
                          if selected_ids is not None and c["id"] not in selected_ids
                          and c["id"] in previous and c["id"] not in {r["id"] for r in rows}]
        args.output.write_text(json.dumps({"model": args.model, "cases": partial}, indent=2))
        print(case["id"], "supported=", case["supported"], "sufficient=", sufficient,
              "citations=", row.get("citation_valid"), "coverage=", row.get("concept_coverage"),
              "fallback=", row.get("fallback_used"), flush=True)

    supported = [r for r in rows if r["supported"]]
    unsupported = [r for r in rows if not r["supported"]]
    mean = lambda key, data: round(sum(float(r.get(key, 0)) for r in data) / len(data), 4)
    summary = {
        "supported_answers": len(supported), "unsupported_questions": len(unsupported),
        "citation_valid_rate": mean("citation_valid", supported),
        "mean_concept_coverage": mean("concept_coverage", supported),
        "mean_citation_label_precision": mean("citation_label_precision", supported),
        "fallback_rate": mean("fallback_used", supported),
        "unsupported_abstention_rate": mean("correct_abstention", unsupported),
        "final_grounded_rate": round(sum(bool(r.get("final_answer"))
            and r.get("concept_coverage") == 1.0
            and r.get("citation_label_precision") == 1.0 for r in supported)
            / len(supported), 4),
    }
    report = {"model": args.model, "summary": summary, "cases": rows}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
