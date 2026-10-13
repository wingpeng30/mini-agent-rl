"""Regression test that keeps provisional cross-approvals short of training."""
import unittest

from scripts.v128_crosscheck_luna_v9_dual_votes import audit


class V9DualVoteCrosscheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_only_citation_valid_cross_approvals_are_counted(self):
        report = self.report
        self.assertEqual(report["state_count"], 55)
        self.assertEqual(report["repair_candidate_states"], 43)
        self.assertEqual(report["nonrepair_states"], 12)
        self.assertEqual(report["deepseek_protocol_invalid_count_under_current_citation_rule"], 12)
        self.assertEqual(report["luna_protocol_invalid_count"], 0)
        self.assertEqual(report["repair_states_both_approve_and_citation_valid"], 33)
        self.assertEqual(report["repair_states_also_in_primary_disagreement_hold"], 1)
        self.assertEqual(report["repair_states_provisionally_clean_of_old_hold"], 32)
        self.assertEqual(report["training_eligible_states"], 0)
        self.assertFalse(report["deepseek_endpoint_identity_verified"])
        self.assertFalse(report["training_allowed"])


if __name__ == "__main__":
    unittest.main()
