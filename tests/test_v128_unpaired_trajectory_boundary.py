"""Ensure unpaired diagnostics and oracle-built traces never become trainable."""
import unittest

from scripts.v128_audit_unpaired_trajectory_boundary import audit


class UnpairedTrajectoryBoundaryTests(unittest.TestCase):
    def test_all_35_unpaired_questions_remain_diagnostic_only(self):
        report = audit()
        self.assertEqual(report["question_count"], 35)
        self.assertEqual(report["triage_dispositions"], {
            "exclude_original": 5, "replay_candidate": 18, "rewrite_question": 12})
        self.assertEqual(report["frozen_structural_rejection_reasons"], {
            "unpairable_in_frozen_construction": 13,
            "no_pair_in_new_shared_corpus": 21,
            "packet_invalid:token_limit": 1})
        self.assertEqual(report["persisted_model_generated_trajectories"], 0)
        self.assertEqual(report["new_complete_pairs_created"], 0)
        self.assertFalse(report["independent_semantic_pair_review_performed"])
        self.assertEqual(report["training_eligible_states"], 0)
        self.assertFalse(report["training_allowed"])


if __name__ == "__main__":
    unittest.main()
