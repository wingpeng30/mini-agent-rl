"""Safety checks for dependency acceptance, not claims of semantic correctness."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.v128_goal_workflow import (
    PLAN, checked_path, digest, encode, evaluate, node, read_bytes, verify_receipt,
)


class GoalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.a = node("A", "A", [], "work", ["criterion"])
        self.b = node("B", "B", ["A"], "work", ["criterion"])
        self.plan = {"nodes": [self.a, self.b]}
        self.blob = b'{"actual_result": 1}'
        self.reader = lambda path: self.blob

    def receipt(self, n, deps=None, status="passed"):
        return {"node": n["id"], "plan_sha256": "plan", "auditor": "test-auditor",
                "dependencies": deps or {}, "status": status,
                "evidence": [{"path": "docs/evidence.json", "sha256": digest(self.blob)}],
                "criteria": {k: {"result": "passed", "explanation": "Checked source",
                                 "evidence": [0]} for k in n["criteria"]},
                "reason": "service unavailable", "resume_condition": "service available"}

    def test_pass_unlocks_only_dependencies_and_no_repeat(self):
        a = encode(self.receipt(self.a))
        states, _, selected = evaluate(self.plan, "plan", {"A": a}, self.reader)
        self.assertEqual((states["A"], states["B"], selected), ("passed", "ready", "B"))
        b = encode(self.receipt(self.b, {"A": digest(a)}))
        self.assertIsNone(evaluate(self.plan, "plan", {"A": a, "B": b}, self.reader)[2])

    def test_changed_evidence_revokes_upstream_and_blocks_downstream(self):
        a = encode(self.receipt(self.a))
        b = encode(self.receipt(self.b, {"A": digest(a)}))
        states, _, selected = evaluate(self.plan, "plan", {"A": a, "B": b}, lambda _: b"changed")
        self.assertEqual(states["B"], "waiting_dependencies")
        self.assertEqual(selected, "A")

    def test_changed_upstream_receipt_invalidates_downstream(self):
        a = encode(self.receipt(self.a))
        b = encode(self.receipt(self.b, {"A": "outdated"}))
        states, _, selected = evaluate(self.plan, "plan", {"A": a, "B": b}, self.reader)
        self.assertEqual((states["B"], selected), ("stale_or_invalid", "B"))

    def test_bool_or_missing_criterion_does_not_pass(self):
        for change in ({"evidence": []}, {"criteria": {}}, {"status": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_receipt(self.a, {**self.receipt(self.a), **change}, "plan", {}, self.reader)

    def test_blocked_and_failed_skip_to_independent_work(self):
        c = node("C", "C", [], "work", ["criterion"])
        for status in ("blocked", "failed"):
            a = encode(self.receipt(self.a, status=status))
            states, _, selected = evaluate({"nodes": [self.a, self.b, c]}, "plan", {"A": a}, self.reader)
            self.assertEqual((states["B"], selected), ("waiting_dependencies", "C"))

    def test_cycles_unknown_dependencies_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({"nodes": [self.b, self.a]}, "plan", {}, self.reader)

    def test_sealed_and_traversal_rejected_without_read(self):
        for path in ("docs/final-test/x.json", "docs/independent100.json", "../outside.json",
                     "data/holdout100/data.json", "C:/secret.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                checked_path(path)

    def test_only_named_historical_reports_are_evidence_allowlisted(self):
        self.assertEqual(checked_path("reports/v122-p3-decision.json").name,
                         "v122-p3-decision.json")
        with self.assertRaises(ValueError):
            checked_path("reports/arbitrary-private-run.jsonl")

    def test_archives_and_archive_members_are_rejected_before_file_read(self):
        for path in ("data/evidence-sft-v124-forensic/candidate.zip",
                     "data/evidence-sft-v124-forensic/candidate.zip/independent100.json"):
            with self.subTest(path=path), patch.object(
                    Path, "read_bytes", side_effect=AssertionError("must not read archive")) as read:
                with self.assertRaisesRegex(ValueError, "archive_paths_rejected_before_read"):
                    read_bytes(path)
                read.assert_not_called()

    def test_real_plan_starts_at_baseline(self):
        states, _, selected = evaluate(PLAN, "plan", {}, self.reader)
        self.assertEqual(selected, "G0")
        self.assertEqual(states["G7"], "waiting_dependencies")


if __name__ == "__main__":
    unittest.main()
