from __future__ import annotations

import unittest

from src.citations import citations_are_complete, claims_are_supported_by_citations
from src.index import normalize_query, method_purpose_score


class CoreRegressionTests(unittest.TestCase):
    def test_normalize_query_removes_markdown_wrapper(self):
        self.assertEqual(
            normalize_query("**What are the main stages of image analysis?**"),
            "What are the main stages of image analysis?",
        )

    def test_citation_completeness_accepts_cited_bullets(self):
        answer = "- Gradient clipping bounds each example's contribution [E1].\n- Noise protects privacy [E2]."
        self.assertTrue(citations_are_complete(answer))

    def test_citation_completeness_rejects_uncited_claim(self):
        self.assertFalse(citations_are_complete("Noise protects privacy."))

    def test_claim_grounding_rejects_unrelated_claim(self):
        evidence = {
            "E1": {"text": "The algorithm clips gradients and adds Gaussian noise to protect privacy."}
        }
        self.assertFalse(
            claims_are_supported_by_citations(
                "The method uses regularization to explain hidden representations [E1].",
                evidence,
            )
        )

    def test_method_purpose_detects_methodological_passage(self):
        methods, purposes = method_purpose_score(
            "At each SGD step we clip each gradient, compute the average, add noise to protect privacy, "
            "and use the privacy accountant to compute privacy loss."
        )
        self.assertGreater(methods, 0)
        self.assertGreater(purposes, 0)


if __name__ == "__main__":
    unittest.main()
