"""Check bounded mechanical feasibility without materializing synthetic answer traces."""
import unittest

from scripts.v128_audit_changed_query_context_feasibility import audit


class ChangedQueryContextFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_only_one_changed_search_has_an_existing_exact_answer_slot(self):
        report = self.report
        self.assertEqual(report["changed_query_actions"], 10)
        self.assertEqual(report["changed_queries_in_v117_repair_candidate_set"], 9)
        self.assertEqual(report["changed_query_groups"], 8)
        self.assertEqual(report["same_group_exact_prefix_immediate_answer_slots"], 1)
        self.assertEqual(report["candidate_rows_materialized"], 0)
        self.assertEqual(len(report["potential_links"]), 1)
        link = report["potential_links"][0]
        self.assertTrue(link["same_action_time_prefix_exact"])
        self.assertEqual(link["answer_context_rebuild_status"],
                         "structural_slot_only_not_materialized")
        self.assertFalse(report["training_allowed"])


if __name__ == "__main__":
    unittest.main()
