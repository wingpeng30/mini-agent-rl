"""Regression tests for strict action-result-answer prefix links."""

import json
import unittest

from scripts.v128_audit_repair_replay_coverage import exact_prefix_link


class ExactPrefixLinkTests(unittest.TestCase):
    def setUp(self):
        self.prefix = [
            {"role": "system", "content": "evidence agent"},
            {"role": "user", "content": "question"},
        ]
        self.action = {"action": "search", "query": "entity relation"}
        self.result = [{"id": "doc-1", "title": "Evidence", "text": "Fact."}]
        self.search_state = {
            "visible_messages": self.prefix,
            "proposed_action": self.action,
            "retrieved_after_search": self.result,
        }

    def answer_state(self):
        messages = self.prefix + [
            {"role": "assistant", "content": json.dumps(self.action)},
            {"role": "tool", "content": json.dumps({"ok": True, "result": self.result})},
        ]
        return {"visible_messages": messages, "proposed_action": {"action": "answer", "answer": "Fact"}}

    def test_accepts_exact_prefix_ending_immediately_before_answer(self):
        self.assertTrue(exact_prefix_link(self.search_state, self.answer_state()))

    def test_rejects_missing_tool_result(self):
        answer = self.answer_state()
        answer["visible_messages"] = answer["visible_messages"][:-1]
        self.assertFalse(exact_prefix_link(self.search_state, answer))

    def test_rejects_changed_retrieval_result(self):
        answer = self.answer_state()
        answer["visible_messages"][-1]["content"] = json.dumps({
            "ok": True, "result": [{"id": "other", "title": "Other", "text": "No."}]
        })
        self.assertFalse(exact_prefix_link(self.search_state, answer))


if __name__ == "__main__":
    unittest.main()
