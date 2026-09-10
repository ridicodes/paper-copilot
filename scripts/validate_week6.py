"""Run the local handoff benchmark; --live calls the running Ollama server."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.index import hybrid_search, is_methodology_question

QUERIES = [
    'What are the main stages of image analysis?',
    'What weaknesses of region-based segmentation techniques are discussed in the paper?',
    'What is pattern recognition and what is its role in computer vision?',
    'What applications of computer vision are mentioned in the paper?',
    'What accuracy does the YOLOv8 model achieve on the COCO dataset?',
    'How do the authors protect private training data?',
    'How do the two papers use machine learning differently?',
    'How can neural networks learn without revealing individual records?',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', default='/tmp/paper-copilot-week6-validation.json')
    args = parser.parse_args()
    namespace = {}
    exec(compile((ROOT / 'app.py').read_text().split('st.set_page_config(')[0],
                 str(ROOT / 'app.py'), 'exec'), namespace)
    report = []
    for query in QUERIES:
        results = hybrid_search(ROOT / 'outputs/library_index', query,
                                k=namespace["COMPARISON_SEARCH_K"])
        sufficient = namespace['evidence_is_sufficient'](results, query)
        record = dict(query=query, sufficient=sufficient)
        if sufficient:
            selected = namespace['select_answer_evidence'](query, results)
            prompt, evidence = namespace['build_answer_prompt'](query, selected)
            record['evidence'] = evidence
            if args.live:
                raw = namespace['ollama_chat'](prompt)
                raw = namespace['normalize_answer'](raw, query)
                if raw.startswith(('Could not connect', 'Ollama timed out', 'Ollama error')):
                    raise RuntimeError(raw)
                valid = namespace['evidence_ids_are_valid'](raw, evidence)
                if is_methodology_question(query) and not namespace['is_cross_document_question'](query):
                    valid = valid and namespace['methodology_answer_is_complete'](raw, selected)
                record.update(raw_answer=raw, citations_complete=namespace['evidence_ids_are_valid'](raw, evidence),
                              answer_valid=valid,
                              fallback_used=not valid and raw != 'Not found in the provided evidence.')
                record['answer'] = (namespace['replace_evidence_ids_with_citations'](raw, evidence)
                                    if valid else raw if raw == 'Not found in the provided evidence.'
                                    else namespace['make_extractive_answer'](query, selected))
        report.append(record)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(query, 'supported=', sufficient, 'complete=', record.get('citations_complete'),
              'fallback=', record.get('fallback_used'), flush=True)
        if 'answer' in record:
            print(record['answer'], flush=True)
    assert not report[4]['sufficient'], 'Unsupported question reached answer generation'
    assert all(item['sufficient'] for i, item in enumerate(report) if i != 4)


if __name__ == '__main__':
    main()
