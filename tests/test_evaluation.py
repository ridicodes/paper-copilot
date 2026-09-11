import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_answers import concept_coverage, labelled
from scripts.evaluate_retrieval import aggregate, evaluate_case, relevant


class EvaluationTests(unittest.TestCase):
    def test_dataset_has_required_categories_and_valid_labels(self):
        cases = json.loads(Path('evaluation/week7_questions.json').read_text())
        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual(len({case['id'] for case in cases}), len(cases))
        kinds = {case['type'] for case in cases}
        self.assertTrue({'fact', 'definition', 'list', 'limitation', 'methodology',
                         'paraphrase', 'comparison', 'unsupported'} <= kinds)
        for case in cases:
            self.assertEqual(bool(case['relevant']), case['supported'])

    def test_relevance_uses_document_and_page(self):
        labels = [{'document': 'a.pdf', 'pages': [2, 3]}]
        self.assertTrue(relevant({'document': 'a.pdf', 'page': 2}, labels))
        self.assertTrue(labelled({'document': 'a.pdf', 'page': 3}, labels))
        self.assertFalse(relevant({'document': 'b.pdf', 'page': 2}, labels))
        self.assertFalse(relevant({'document': 'a.pdf', 'page': 4}, labels))

    def test_concept_coverage_accepts_alternatives(self):
        groups = [['clip', 'bound'], ['noise'], ['accountant', 'privacy loss']]
        self.assertEqual(concept_coverage('Bound gradients, add noise, track privacy loss.', groups), 1.0)
        self.assertEqual(concept_coverage('Add noise.', groups), 0.3333)

    def test_aggregate_metrics(self):
        rows = [
            dict(supported=True, type='fact', hit_at_1=True, hit_at_3=True,
                 hit_at_5=True, first_relevant_rank=1, sufficient=True,
                 all_documents_at_5=True, noise_top_5=0),
            dict(supported=False, type='unsupported', hit_at_1=False, hit_at_3=False,
                 hit_at_5=False, first_relevant_rank=None, sufficient=False,
                 correct_rejection=True, all_documents_at_5=True, noise_top_5=0),
        ]
        summary = aggregate(rows)
        self.assertEqual(summary['hit_at_1'], 1.0)
        self.assertEqual(summary['unsupported_rejection'], 1.0)


if __name__ == '__main__':
    unittest.main()
