import unittest
from src.citations import (
    citations_are_complete,
    comparison_citations_cover_documents,
    normalize_answer,
    repair_comparison_synthesis_citations,
)


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

    def test_comparison_repair_is_narrow_and_still_validated(self):
        evidence = {
            'E1': {'document': 'privacy.pdf', 'page': 1},
            'E2': {'document': 'vision.pdf', 'page': 6},
        }
        answer = (
            '- Paper A protects neural-network training data. [E1]\n'
            '- Paper B analyzes images to recognize objects. [E2]\n'
            'The approaches differ because Paper A protects training while '
            'Paper B interprets images.'
        )
        repaired = repair_comparison_synthesis_citations(answer, evidence)
        self.assertTrue(repaired.endswith('[E1] [E2]'))
        self.assertTrue(citations_are_complete(repaired))
        self.assertTrue(comparison_citations_cover_documents(repaired, evidence))
        self.assertFalse(comparison_citations_cover_documents(
            '- Paper A protects training data. [E1]', evidence,
        ))

        unrelated = answer.replace(
            'The approaches differ because Paper A protects training while Paper B interprets images.',
            'This proves both systems are universally reliable.',
        )
        self.assertEqual(repair_comparison_synthesis_citations(unrelated, evidence), unrelated)
        self.assertFalse(citations_are_complete(unrelated))

        one_sided = answer.replace('[E2]', '[E1]')
        self.assertEqual(repair_comparison_synthesis_citations(one_sided, evidence), one_sided)
        self.assertFalse(citations_are_complete(one_sided))
