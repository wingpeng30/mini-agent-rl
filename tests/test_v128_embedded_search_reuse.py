"""Regression checks for exact-state overlap handling of embedded searches."""

import unittest

from scripts.v128_audit_embedded_search_reuse import audit


class EmbeddedSearchReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_recounts_embedded_searches_and_existing_target_overlaps(self):
        counts = self.report["counts"]
        self.assertEqual(counts["embedded_successful_search_tool_pairs"], 162)
        self.assertEqual(counts["unique_question_prefix_action_states"], 162)
        self.assertEqual(counts["exact_overlap_frozen_search_targets"], 14)
        self.assertEqual(counts["exact_overlap_retained_core_search_targets"], 12)
        self.assertEqual(counts["exact_overlap_nonretained_records_marked_repair_candidate"], 2)
        self.assertEqual(counts["exact_duplicate_occurrences_beyond_unique"], 0)

    def test_ten_extracted_candidates_are_filtered_without_semantic_promotion(self):
        counts = self.report["counts"]
        self.assertEqual(counts["previously_extracted_action_result_citation_candidates"], 10)
        self.assertEqual(counts["extracted_candidates_exact_duplicate_current_retained_train_states"], 3)
        self.assertEqual(counts["extracted_candidates_exact_duplicate_existing_frozen_search_targets"], 4)
        self.assertEqual(counts["extracted_candidates_novel_to_frozen_search_target_states"], 6)
        self.assertEqual(counts["extracted_candidates_both_source_rows_in_retained_core"], 10)
        self.assertEqual(counts["extracted_candidates_exact_action_result_answer_prefix_links"], 10)
        self.assertEqual(counts["extracted_candidates_new_semantic_reviews"], 0)
        self.assertEqual(counts["extracted_candidates_export_eligible"], 0)
        self.assertFalse(self.report["training_allowed"])
        self.assertFalse(self.report["data_ready_for_experiment"])


if __name__ == "__main__":
    unittest.main()
