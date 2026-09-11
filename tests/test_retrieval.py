import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src import index
from src.evidence import evidence_is_sufficient, query_anchors


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Deliberately duplicate chunk IDs and page numbers across documents.
        for name, texts in [('a', ['Neural networks protect private training data.',
                                   'Image analysis starts with image formation.']),
                            ('b', ['Visual recognition classifies images.',
                                   'Noise protects the privacy of training records.'])]:
            (self.root / f'{name}.json').write_text(json.dumps([
                dict(chunk_id=str(i), page=1, text=text) for i, text in enumerate(texts)
            ]))
        self.vectors = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
        with patch.object(index, 'build_embeddings', return_value=self.vectors):
            index.build_index([self.root / 'a.json', self.root / 'b.json'], self.root / 'idx')

    def test_round_trip_and_legacy_bm25(self):
        bm25, chunks = index.load_index(self.root / 'idx')
        self.assertEqual(len(chunks), 4)
        self.assertEqual(bm25.corpus_size, 4)
        (self.root / 'idx' / 'embeddings.npy').unlink()
        self.assertTrue(index.search(self.root / 'idx', 'training'))
        with self.assertRaises(FileNotFoundError):
            index.load_embeddings(self.root / 'idx')

    def test_hybrid_identity_diversity_and_rank_diagnostics(self):
        with patch.object(index, 'build_embeddings', return_value=np.array([[1, 0]])):
            results = index.hybrid_search(self.root / 'idx', 'private training', k=4,
                                          max_per_page=1)
        self.assertEqual({r['document'] for r in results}, {'a.pdf', 'b.pdf'})
        self.assertEqual(len(results), 2)
        for r in results:
            expected = sum(1 / (60 + r[key]) for key in ['bm25_rank', 'semantic_rank']
                           if r[key] is not None)
            self.assertAlmostEqual(r['rrf_score'], expected)
            self.assertEqual(r['text'], index.load_index(self.root / 'idx')[1][r['chunk_index']]['text'])

    def test_corrupt_embeddings_are_rejected(self):
        np.save(self.root / 'idx' / 'embeddings.npy', np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            index.load_embeddings(self.root / 'idx')

    def test_blank_queries_and_zero_k(self):
        for method in [index.search, index.semantic_search, index.hybrid_search]:
            self.assertEqual(method('missing', ''), [])
            self.assertEqual(method('missing', 'question', k=0), [])

    def test_semantic_gate_and_missing_entities(self):
        result = dict(document='a.pdf', text='Privacy protects training records.',
                      retrieval_method='hybrid', coverage=0.1, semantic_score=0.65,
                      bm25_score=1, noise_penalty=0)
        self.assertTrue(evidence_is_sufficient('How are personal examples concealed?', [result]))
        self.assertFalse(evidence_is_sufficient('What accuracy does YOLOv8 achieve on COCO?', [result]))
        self.assertFalse(evidence_is_sufficient('Compare both papers', [result], comparison=True))
        self.assertFalse(evidence_is_sufficient('Privacy?', [dict(result, noise_penalty=4)]))
        self.assertFalse(evidence_is_sufficient('Unknown topic?', [dict(result, semantic_score=.1)]))

    def test_comparison_detection_uses_whole_words(self):
        self.assertFalse(index.is_comparison_question(
            'What results are reported for different privacy budgets?'))
        self.assertFalse(index.is_comparison_question('What is differential privacy?'))
        self.assertTrue(index.is_comparison_question(
            'How do the two papers use machine learning differently?'))

    def test_query_normalization_removes_pasted_markdown(self):
        decorated = 'Ask: > **How do the two papers use machine learning differently?**'
        self.assertEqual(
            index.normalize_query(decorated),
            'How do the two papers use machine learning differently?',
        )
        self.assertNotIn('ask', index.query_terms(decorated))

    def test_currency_is_an_explicit_anchor(self):
        self.assertIn('dollars', query_anchors('What was the training cost in US dollars?'))

    def test_citation_metadata_is_preserved(self):
        # Load pure app helpers without running the Streamlit page.
        tree = ast.parse(Path('app.py').read_text())
        names = {'build_evidence_map', 'replace_evidence_ids_with_citations',
                 'extract_evidence_ids', 'evidence_ids_are_valid',
                 'citation_text', 'document_display_name'}
        helpers = ast.Module(body=[node for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name in names],
                             type_ignores=[])
        import re
        from src.citations import citations_are_complete
        namespace = {'Path': Path, 're': re, 'citations_are_complete': citations_are_complete}
        exec(compile(helpers, 'app.py', 'exec'), namespace)
        evidence = namespace['build_evidence_map']([
            dict(document='a.pdf', page=2, text='First'),
            dict(document='b.pdf', page=2, text='Second'),
        ])
        self.assertTrue(namespace['evidence_ids_are_valid']('Claim [E1] [E2]', evidence))
        self.assertFalse(namespace['evidence_ids_are_valid']('Claim [E9]', evidence))
        answer = namespace['replace_evidence_ids_with_citations']('Claim [E1] [E2]', evidence)
        self.assertIn('[a, p. 2]', answer)
        self.assertIn('[b, p. 2]', answer)


if __name__ == '__main__':
    unittest.main()
