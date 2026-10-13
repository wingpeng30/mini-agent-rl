"""Regression test for the bounded historical runner serializer audit."""
import unittest

from scripts.v128_audit_repair_runner_serialization import audit


class RepairRunnerSerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_available_runners_do_not_establish_v117_serializer(self):
        report = self.report
        self.assertEqual(report["historical_answer_message_count"], 6)
        self.assertTrue(report["historical_tool_message_shapes"])
        self.assertTrue(all("searched_queries" not in shape["top_level_keys"]
                            for shape in report["historical_tool_message_shapes"]))
        self.assertTrue(report["current_runner_adds_searched_queries"])
        self.assertTrue(report["comparison_runner_adds_searched_queries"])
        self.assertFalse(report["v117_runtime_commit_bound"])
        self.assertFalse(report["answer_context_materialized"])
        self.assertFalse(report["training_allowed"])


if __name__ == "__main__":
    unittest.main()
