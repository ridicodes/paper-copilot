import unittest
from src.citations import (
    citations_are_complete,
    claims_are_supported_by_citations,
    comparison_citations_cover_documents,
    normalize_answer,
    normalize_comparison_answer,
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

        trailing_only = (
            '- Paper A: Objective is privacy. Method is private training. [E1]\n'
            '- Paper B: Objective is recognition. Method is image analysis. [E2]'
        )
        repaired_bullets = repair_comparison_synthesis_citations(
            trailing_only, evidence)
        self.assertTrue(citations_are_complete(repaired_bullets))
        self.assertEqual(repaired_bullets.count('[E1]'), 2)
        self.assertEqual(repaired_bullets.count('[E2]'), 2)

    def test_comparison_normalization_removes_only_known_wrappers(self):
        wrapped = (
            'Based on the provided evidence, here is the answer to the question:\n\n'
            '**Main objective of each paper:**\n'
            '- Paper A — Objective: privacy [E1]. Method: private training [E1].\n'
            '- Paper B — Objective: image analysis [E2]. Method: vision [E2].\n'
            'Note: The evidence does not provide more implementation detail.'
        )
        normalized = normalize_comparison_answer(wrapped)
        self.assertNotIn('here is the answer', normalized.lower())
        self.assertNotIn('Main objective', normalized)
        self.assertNotIn('Note:', normalized)
        self.assertTrue(citations_are_complete(normalized))

        uncited_claim = normalized + '\nBoth methods are universally reliable.'
        self.assertIn('universally reliable',
                      normalize_comparison_answer(uncited_claim))
        self.assertFalse(citations_are_complete(
            normalize_comparison_answer(uncited_claim)))

    def test_claims_must_match_the_exact_cited_evidence(self):
        evidence = {
            'E1': {
                'document': 'privacy.pdf',
                'grounding_text': (
                    'We combine machine learning with advanced privacy-preserving '
                    'mechanisms, training neural networks within a privacy budget.'
                ),
            },
            'E2': {
                'document': 'vision.pdf',
                'grounding_text': (
                    'Machine learning and image processing recognize patterns '
                    'of increasingly diverse objects.'
                ),
            },
        }
        grounded = (
            '- Paper A trains neural networks with privacy-preserving mechanisms '
            'within a privacy budget. [E1]\n'
            '- Paper B uses machine learning and image processing to recognize '
            'object patterns. [E2]'
        )
        self.assertTrue(claims_are_supported_by_citations(grounded, evidence))

        misattributed = grounded.replace(
            'trains neural networks with privacy-preserving mechanisms within a privacy budget',
            'uses regularization to avoid overfitting and explain internal representations',
        )
        self.assertFalse(claims_are_supported_by_citations(misattributed, evidence))
