"""Optional integration checks using the handoff's local PDFs and cached model."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from src.index import hybrid_search
from src.evidence import evidence_is_sufficient


@unittest.skipUnless(os.environ.get('PAPER_COPILOT_LIBRARY_TESTS') == '1',
                     'Set PAPER_COPILOT_LIBRARY_TESTS=1 with the handoff library installed')
class LibraryTests(unittest.TestCase):
    def test_retrieval_regressions(self):
        cases = [
            ('What are the main stages of image analysis?', 2, 'image formation'),
            ('What weaknesses of region-based segmentation techniques are discussed in the paper?', 5, 'weakness'),
            ('How do the authors protect private training data?', 3, 'SGD'),
        ]
        for query, page, phrase in cases:
            with self.subTest(query=query):
                results = hybrid_search('outputs/library_index', query)
                self.assertEqual(results[0]['page'], page)
                self.assertIn(phrase.lower(), results[0]['text'].lower())
                self.assertTrue(evidence_is_sufficient(query, results))
        for query in [
            'What accuracy does the YOLOv8 model achieve on the COCO dataset?',
            'What does the paper say about quantum entanglement?',
            'What accuracy does the model achieve on an unseen Mars rover dataset?',
        ]:
            self.assertFalse(evidence_is_sufficient(
                query, hybrid_search('outputs/library_index', query)), query)

    def test_streamlit_answer_flow(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'),
                                default_timeout=45).run()
        app.session_state['idx_dir'] = 'outputs/library_index'
        app.session_state['pdf_paths'] = [str(path) for path in Path('data').glob('*.pdf')]
        app.run()
        app.text_input[0].set_value('What accuracy does the YOLOv8 model achieve on the COCO dataset?')
        for mode in ['BM25', 'Semantic', 'Hybrid']:
            app.selectbox[0].set_value(mode)
            with patch('src.llm.ollama_chat') as llm:
                app.button[1].click().run()
                llm.assert_not_called()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state['answer'])
        for query in [
            'How do the two papers use machine learning differently?',
            'How can neural networks learn without revealing individual records?',
        ]:
            app.text_input[0].set_value(query)
            with patch('src.llm.ollama_chat', return_value='Supported statement. [E1]') as llm:
                app.button[1].click().run()
                llm.assert_called_once()
            self.assertFalse(app.exception)
            self.assertIn(', p. ', app.session_state['answer'])
            self.assertNotIn('[E1]', app.session_state['answer'])

    def test_methodology_prompt_keeps_algorithm_steps(self):
        namespace = {}
        source = Path('app.py').read_text().split('st.set_page_config(')[0]
        exec(compile(source, 'app.py', 'exec'), namespace)
        query = 'How can neural networks learn without revealing individual records?'
        results = hybrid_search('outputs/library_index', query, k=20)
        selected = namespace['select_answer_evidence'](query, results)
        prompt, _ = namespace['build_answer_prompt'](query, selected)
        self.assertEqual(selected[0]['page'], 3)
        for phrase in ['clip', 'compute the average', 'add noise', 'privacy accountant']:
            self.assertIn(phrase, prompt)
        self.assertFalse(namespace['methodology_answer_is_complete']('They add noise. [E1]', selected))
        fallback = namespace['make_extractive_answer'](query, selected)
        for phrase in ['clip', 'compute the average', 'add noise', 'privacy accountant']:
            self.assertIn(phrase, fallback)

    def test_direct_definition_and_sensor_evidence_selection(self):
        namespace = {}
        source = Path('app.py').read_text().split('st.set_page_config(')[0]
        exec(compile(source, 'app.py', 'exec'), namespace)
        cases = [
            ('What does differential privacy protect in the training dataset?', 2, 'one record'),
        ]
        for query, page, phrase in cases:
            results = hybrid_search('outputs/library_index', query, k=20)
            selected = namespace['select_answer_evidence'](query, results)
            self.assertEqual(selected[0]['page'], page)
            self.assertIn(phrase, selected[0]['text'].lower())

    def test_accuracy_question_selects_reported_values(self):
        namespace = {}
        source = Path('app.py').read_text().split('st.set_page_config(')[0]
        exec(compile(source, 'app.py', 'exec'), namespace)
        query = 'What MNIST test accuracies are reported for different privacy budgets?'
        selected = namespace['select_answer_evidence'](
            query, hybrid_search('outputs/library_index', query, k=20))
        self.assertEqual(selected[0]['page'], 6)
        for value in ('90%', '95%', '97%'):
            self.assertIn(value, selected[0]['text'])

    def test_concise_direct_evidence_selection(self):
        namespace = {}
        source = Path('app.py').read_text().split('st.set_page_config(')[0]
        exec(compile(source, 'app.py', 'exec'), namespace)
        cases = [
            ('What weaknesses of region-based segmentation techniques are discussed?', [5]),
            ('Why are edge detection and segmentation important in computer vision applications?', [3]),
            ('Which datasets are used to evaluate the private neural networks?', [6]),
        ]
        for query, pages in cases:
            selected = namespace['select_answer_evidence'](
                query, hybrid_search('outputs/library_index', query, k=20))
            self.assertEqual([result['page'] for result in selected], pages)

        comparison = 'How do the two papers use machine learning differently?'
        selected = namespace['select_answer_evidence'](
            comparison, hybrid_search('outputs/library_index', comparison, k=20))
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({result['document'] for result in selected}), 2)

        conceptual = (
            'How do the two papers use machine learning differently, '
            'and what problem does each paper aim to solve?'
        )
        selected = namespace['select_answer_evidence'](
            conceptual, hybrid_search('outputs/library_index', conceptual, k=20))
        self.assertEqual(len({result['document'] for result in selected}), 2)
        privacy = ' '.join(result['text'].lower() for result in selected
                           if result['document'].startswith('1607'))
        vision = ' '.join(result['text'].lower() for result in selected
                          if result['document'].startswith('Computer_Vision'))
        self.assertIn('privacy', privacy)
        self.assertRegex(vision, r'analy[sz]e images|predict or detect|recognize patterns')
        fallback = namespace['make_extractive_answer'](conceptual, selected)
        self.assertIn('privacy', fallback.lower())
        self.assertRegex(fallback.lower(), r'analy[sz]e images|predict or detect|recognize')
        visible = namespace['comparison_first_results'](
            conceptual, hybrid_search('outputs/library_index', conceptual, k=20))
        self.assertEqual(len({result['document'] for result in visible[:2]}), 2)
        self.assertIn('privacy', visible[0]['text'].lower())
        self.assertRegex(visible[1]['text'].lower(),
                         r'analy[sz]e images|predict or detect|recognize patterns')

        generic = (
            'What challenge does each paper try to address, '
            'and how do their approaches differ?'
        )
        generic_results = namespace['comparison_first_results'](
            generic, hybrid_search('outputs/library_index', generic, k=20))
        self.assertEqual(len({result['document'] for result in generic_results[:2]}), 2)
        self.assertTrue(namespace['evidence_is_sufficient'](generic_results, generic))

        image_comparison = 'Compare the role of images in the two papers.'
        selected = namespace['select_answer_evidence'](
            image_comparison, hybrid_search('outputs/library_index', image_comparison, k=20))
        self.assertEqual(len({result['document'] for result in selected}), 2)
        self.assertTrue(all('image' in result['text'].lower() for result in selected))

    def test_page_viewer_and_mode_changes(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'),
                                default_timeout=45).run()
        app.session_state['idx_dir'] = 'outputs/library_index'
        app.session_state['pdf_paths'] = [str(path) for path in Path('data').glob('*.pdf')]
        app.run()
        for question, document_prefix in [
            ('What are the main stages of image analysis?', 'Computer'),
            ('How do the authors protect private training data?', '1607'),
        ]:
            app.text_input[0].set_value(question)
            app.button[0].click().run()
            next(button for button in app.button if button.label == 'View page').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state['view_document'].startswith(document_prefix))
            self.assertTrue(app.get('image'))
        app.selectbox[0].set_value('BM25')
        with patch('src.llm.ollama_chat', return_value='Supported. [E1] Uncited conclusion.'):
            app.button[1].click().run()
        self.assertEqual(app.session_state['last_retrieval_mode'], 'BM25')
        self.assertNotIn('Uncited conclusion', app.session_state['answer'])
        self.assertIn(', p. ', app.session_state['answer'])
        self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
