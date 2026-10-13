"""Regression checks for ID-based v117 repair inventory accounting."""
import unittest

from scripts.v128_audit_repairs import audit


class RepairInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_v117_repair_candidate_ids_reconcile_to_snapshot(self):
        counts = self.report["counts"]
        self.assertEqual(counts["v117_repair_candidates_backlog_component"], 61)
        self.assertEqual(counts["v117_repair_candidates_cpu_reconciliation_component"], 46)
        self.assertEqual(counts["v117_repair_candidate_components_overlap"], 0)
        self.assertEqual(counts["v117_repair_candidates_reconstructed"], 107)
        self.assertTrue(counts["v117_repair_candidate_count_matches_snapshot"])
        self.assertEqual(len(self.report["v117_repair_candidate_record_ids"]), 107)

    def test_other_ledgers_are_deduplicated_without_semantic_promotion(self):
        counts = self.report["counts"]
        self.assertEqual(counts["quarantine_states"], 120)
        self.assertEqual(counts["unresolved_v117_backlog_states"], 63)
        self.assertEqual(counts["cpu_reconciliation_states"], 61)
        self.assertEqual(counts["union_distinct_record_ids"], 234)
        self.assertEqual(counts["unpaired_questions"], 35)
        self.assertEqual(counts["semantic_double_approved_repair_states"], 0)
        self.assertFalse(self.report["training_allowed"])
        self.assertFalse(self.report["data_ready_for_experiment"])


if __name__ == "__main__":
    unittest.main()
