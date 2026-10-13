from __future__ import annotations

import unittest

from scripts.v130_close_g2_dispositions import build, classify


class G2DispositionClosureTests(unittest.TestCase):
    def test_current_repair_inventory_is_partitioned_without_promoting_labels(self):
        report = build()
        self.assertEqual(report["repair_candidate_count"], 107)
        self.assertEqual(report["unmaterialized_excluded"], 61)
        self.assertEqual(report["changed_query_context_excluded"], 9)
        self.assertEqual(report["normalized_overlap_excluded"], 1)
        self.assertEqual(report["exclusion_intersection_count"], 1)
        self.assertEqual(report["future_semantic_review_only"], 37)
        unique_excluded = (report["unmaterialized_excluded"]
                           + report["changed_query_context_excluded"]
                           + report["normalized_overlap_excluded"]
                           - report["exclusion_intersection_count"])
        self.assertEqual(unique_excluded, 70)
        self.assertEqual(report["semantic_double_approvals_added"], 0)
        self.assertEqual(report["strict_complete_pairs_added"], 0)
        self.assertTrue(all(not row["export_eligible"] for row in report["rows"]))
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["data_ready_for_experiment"])

    def test_partition_mismatch_stops_instead_of_filling_from_other_ids(self):
        with self.assertRaisesRegex(ValueError, "materialized_unmaterialized_partition_mismatch"):
            # Counts deliberately satisfy the fixed cohort totals while one candidate vanishes.
            repair = {f"r{i}" for i in range(107)}
            materialized = {f"r{i}" for i in range(46)}
            unmaterialized = {f"r{i}" for i in range(45, 106)}
            coverage = {key: {"within_limits": True, "exact_state_duplicate_in_revised_pool": False,
                              "exact_state_duplicate_in_current_training_pool": False,
                              "source_row_sha256": "a", "revised_row_sha256": "b",
                              "revised_visible_prefix_sha256": "c", "action_target_sha256": "d",
                              "semantic_review": "not_run"} for key in materialized}
            classify(repair, materialized, unmaterialized, set(), set(), coverage)


if __name__ == "__main__":
    unittest.main()
