from __future__ import annotations

import unittest

from scripts.v130_audit_v123_reuse_v129 import _for_validation, audit


class V123ReuseAuditTests(unittest.TestCase):
    def test_answer_null_normalization_is_validation_only(self):
        packet = {"case_id": "x", "proposed_action": {"action": "answer"},
                  "retrieved_after_search": None, "packet_sha256": "original"}
        checked = _for_validation(packet)
        self.assertEqual(checked["retrieved_after_search"], [])
        self.assertIsNone(packet["retrieved_after_search"])
        self.assertEqual(packet["packet_sha256"], "original")

    def test_search_null_is_not_normalized(self):
        packet = {"proposed_action": {"action": "search"}, "retrieved_after_search": None}
        self.assertIsNone(_for_validation(packet)["retrieved_after_search"])

    def test_exact_v123_reuse_recount_is_stable_and_not_data_clearance(self):
        report = audit()
        self.assertEqual(report["mapped_states"], 42)
        self.assertEqual(report["complete_dual_approved_pairs_under_v129"], 19)
        self.assertEqual(report["risk_intersection_states_in_complete_pairs"],
                         {"relation": 13, "time": 6, "scope": 2, "measurement": 2})
        self.assertFalse(report["data_ready_for_experiment"])
        self.assertFalse(report["training_allowed"])


if __name__ == "__main__":
    unittest.main()
