"""Regression test for the one-time internal candidate-pool stopping proof."""
import unittest

from scripts.v128_audit_v123_one_time_budget import audit


class OneTimeBudgetAuditTests(unittest.TestCase):
    def test_budget_and_saved_vote_upper_bound_are_both_binding(self):
        report = audit()
        pool, upper = report["candidate_pool"], report["saved_vote_upper_bounds"]
        self.assertEqual(pool["screened_source_questions"], 5000)
        self.assertEqual(pool["new_questions"], 100)
        self.assertEqual(pool["combined_questions"], 145)
        self.assertEqual(pool["old_new_id_overlap"], 0)
        self.assertEqual(upper["possible_complete_pairs"], 49)
        self.assertEqual(upper["possible_scope_states"], 2)
        self.assertEqual(upper["possible_measurement_states"], 4)
        self.assertFalse(report["second_pool_or_reseed_authorized"])
        self.assertFalse(report["semantic_unusability_proven"])
        self.assertFalse(report["data_ready_for_experiment"])
        self.assertFalse(report["training_allowed"])


if __name__ == "__main__":
    unittest.main()
