"""Checks the evidence-only holds for unrevised v117 candidates."""

import unittest

from scripts.v128_classify_unmaterialized_repairs import audit


class UnmaterializedRepairDispositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_every_unmaterialized_repair_candidate_has_one_specific_hold(self):
        counts = self.report["counts"]
        rows = self.report["dispositions"]
        self.assertEqual(counts["backlog_repair_candidates"], 61)
        self.assertEqual(counts["unique_record_ids"], 61)
        self.assertEqual(counts["unique_normalized_question_groups"], 30)
        self.assertEqual(len({row["record_id"] for row in rows}), 61)
        self.assertTrue(all(row["current_hold_reason"] for row in rows))
        self.assertTrue(all(not row["in_existing_materialized_revision"] for row in rows))
        self.assertTrue(all(not row["in_existing_replay_input"] for row in rows))

    def test_holds_do_not_promote_proposals_or_authorize_export(self):
        self.assertEqual(self.report["counts"]["semantic_approvals"], 0)
        self.assertEqual(self.report["counts"]["export_eligible"], 0)
        self.assertFalse(self.report["training_allowed"])
        self.assertFalse(self.report["data_ready_for_experiment"])
        self.assertTrue(all(row["semantic_review"] == "not_run_or_not_verified"
                            for row in self.report["dispositions"]))


if __name__ == "__main__":
    unittest.main()
