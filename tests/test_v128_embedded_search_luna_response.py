"""Ensure a single blind diagnostic can never be promoted to dual approval."""
import unittest

from scripts.v128_validate_embedded_search_luna_response import validate_response


class EmbeddedSearchLunaResponseTests(unittest.TestCase):
    def test_response_votes_are_single_review_only_and_case04_gap_is_visible(self):
        report = validate_response()
        self.assertEqual(report["mechanically_valid_votes"], 6)
        self.assertEqual(report["single_vote_approvals"], 5)
        self.assertEqual(report["single_vote_rejections"], 1)
        self.assertEqual(report["review_prompt_prior_context_citation_compliant_votes"], 5)
        self.assertEqual(report["double_review_approvals"], 0)
        self.assertEqual(report["training_eligible_states"], 0)
        self.assertFalse(report["training_allowed"])
        discrepancy = report["manual_discrepancy"]
        self.assertTrue(discrepancy["reviewer_vote_preserved"])
        self.assertTrue(discrepancy["reviewer_rationale_conflicts_with_visible_evidence"])
        self.assertFalse(discrepancy["review_prompt_citation_requirement_met"])
        self.assertTrue(discrepancy["frozen_validator_missed_rejection_citation_requirement"])
        self.assertFalse(discrepancy["re_review_or_retry_requested"])


if __name__ == "__main__":
    unittest.main()
