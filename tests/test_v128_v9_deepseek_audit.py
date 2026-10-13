"""Guard the hash-bound recount of historical v9 DeepSeek review records."""
import unittest

from scripts.v128_audit_v9_deepseek_logs import audit


class V9DeepSeekAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_saved_single_votes_are_not_mistaken_for_dual_review(self):
        report = self.report
        self.assertEqual(report["batch_count"], 3)
        self.assertEqual(report["reviewed_states"], 55)
        self.assertEqual(report["single_reviewer_decisions"], {"approve": 49, "reject": 6})
        self.assertEqual(report["dual_review_approvals"], 0)
        self.assertFalse(report["training_allowed"])

    def test_reviewed_state_membership_and_v9_outstanding_holds_are_crosswalked(self):
        report = self.report
        self.assertEqual(report["source_states_intersecting_v117_repair_candidates"], 43)
        self.assertEqual(report["source_states_outside_v117_repair_candidates"], 12)
        self.assertEqual(report["source_states_by_g2_disposition"], {
            "quarantine_pending_repair": 8,
            "retain_candidate": 4,
            "v117_repair_candidate": 43,
        })
        self.assertEqual(report["batch02_primary_disagreement_states_bound_to_votes"], 7)
        self.assertEqual(report["batch02_primary_disagreement_states_still_held"], 7)
        self.assertEqual(report["batch02_disagreement_states_by_g2_disposition"], {
            "quarantine_pending_repair": 2, "v117_repair_candidate": 5,
        })
        self.assertEqual(report[
            "nonrepair_v9_states_already_exactly_duplicating_current_training_input"], 1)

    def test_validator_drift_and_stale_manifest_flags_remain_visible(self):
        report = self.report
        self.assertEqual(report["manifest_attempt_flag_stale_count"], 3)
        self.assertEqual(report["initial_outputs_passing_current_validator"], 2)
        self.assertTrue(report["citation_corrected_batch01_passes_current_validator"])
        self.assertFalse(report["endpoint_host_logged_in_v9_request_journal"])
        self.assertEqual(report["citation_correction"]["reviewed_states"], 20)
        self.assertTrue(report["citation_correction"]["decisions_and_rationales_unchanged"])


if __name__ == "__main__":
    unittest.main()
