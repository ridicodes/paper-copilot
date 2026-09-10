import unittest
from src.citations import citations_are_complete, normalize_answer


class CitationTests(unittest.TestCase):
    def test_complete_sentences(self):
        for text in ['First claim. [E1] Second claim. [E2]',
                     '- First claim [E1].\n- Second claim [E2] [E3].',
                     'The accuracy is 67.5% [E1].']:
            self.assertTrue(citations_are_complete(text), text)

    def test_incomplete_sentences(self):
        for text in ['Claim [E1]. Uncited conclusion.',
                     'Uncited claim. Supported claim [E1].',
                     '- Cited [E1]\n- Uncited',
                     'Claim [E1] and another uncited assertion.',
                     '# Uncited factual heading\nClaim [E1].', '', '[E1]']:
            self.assertFalse(citations_are_complete(text), text)

    def test_prefaces_do_not_hide_claims(self):
        question = 'What are the main stages of image analysis?'
        answer = question + '\nThe main stages of image analysis are:\n- Formation [E1]'
        self.assertEqual(normalize_answer(answer, question), '- Formation [E1]')
        self.assertFalse(citations_are_complete(normalize_answer(
            'The model achieves perfect accuracy:\nClaim [E1]', question)))
        self.assertFalse(citations_are_complete('Claim [E1]. another uncited claim.'))
