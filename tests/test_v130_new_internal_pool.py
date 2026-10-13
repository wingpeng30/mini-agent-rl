from __future__ import annotations

import unittest

from scripts.v130_build_new_internal_pool import _followups, _require_method_ready, _scope_hint
from scripts.v130_diagnose_linked_title_policy import linked_title_candidates


class TinyIndex:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, top_k=3):
        self.queries.append(query)
        return self.results.get(query, [])[:top_k]


class NewPoolPolicyTests(unittest.TestCase):
    def test_development_audited_method_is_explicitly_enabled(self):
        _require_method_ready()

    def test_followup_uses_only_an_exact_title_mentioned_in_visible_text(self):
        visible = [{"id": "d1", "title": "First Page",
                    "text": "The article explicitly mentions Bridge Page as its source."}]
        candidates = linked_title_candidates("Question asks about the First Page?", visible,
                                             {"bridge page": "Bridge Page", "first page": "First Page"}, set())
        self.assertEqual(candidates, ["Bridge Page"])

    def test_linked_followup_must_add_a_globally_unseen_document(self):
        first = [{"id": "d1", "title": "First Page",
                  "text": "It mentions Bridge Page, which is another article."}]
        index = TinyIndex({"Bridge Page": [{"id": "d1", "title": "First Page", "text": "old"},
                                            {"id": "d2", "title": "Bridge Page", "text": "new"}]})
        result = _followups("Who founded First Page?", first, index,
                            [{"id": "d1", "title": "First Page"}, {"id": "d2", "title": "Bridge Page"}])
        self.assertEqual(result[0][0], "Bridge Page")
        self.assertIn("d2", [doc["id"] for doc in result[0][1]])

    def test_followup_returns_none_when_no_visible_title_adds_document(self):
        first = [{"id": "d1", "title": "First Page", "text": "No linked title here."}]
        index = TinyIndex({})
        self.assertEqual(_followups("First Page question?", first, index,
                                   [{"id": "d1", "title": "First Page"}]), [])

    def test_scope_hint_is_only_a_schedule_flag(self):
        self.assertTrue(_scope_hint("Which city is larger, A or B?"))
        self.assertFalse(_scope_hint("Who directed Film X?"))


if __name__ == "__main__":
    unittest.main()
