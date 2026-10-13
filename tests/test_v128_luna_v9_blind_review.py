"""Check coverage, exact citation binding, and non-promotion of Luna's v9 review."""
import unittest

from scripts.v128_validate_luna_v9_blind_review import audit


class LunaV9BlindReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_complete_coverage_and_repair_inventory_crosswalk(self):
        report = self.report
        self.assertEqual(report["expected_state_count"], 55)
        self.assertEqual(report["reviewed_state_count"], 55)
        self.assertEqual(report["unique_source_record_count"], 55)
        self.assertEqual(report["repair_candidate_state_count"], 43)
        self.assertEqual(report["nonrepair_state_count"], 12)

    def test_citation_validation_does_not_promote_votes(self):
        report = self.report
        self.assertGreater(report["mechanically_valid_review_count"], 0)
        self.assertEqual(report["mechanically_valid_review_count"] +
                         report["mechanically_invalid_count"], 55)
        self.assertFalse(report["semantic_double_review_granted"])
        self.assertFalse(report["training_allowed"])


if __name__ == "__main__":
    unittest.main()
