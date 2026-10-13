import unittest

from scripts.v128_audit_primary import validate_citations


def row(action="answer"):
    return {"proposed_action": {"action": action},
            "before_action": [{"id": "prior", "text": "The known fact is 1995."}]}


def review(stage="before_action", quote="The known fact is 1995.", document_id="prior"):
    return {"citations": [{"stage": stage, "quote": quote, "document_id": document_id}]}


class PrimaryAuditTests(unittest.TestCase):
    def test_rejects_whitespace_quote_that_empty_substring_checks_accept(self):
        with self.assertRaisesRegex(ValueError, "citation_quote_empty"):
            validate_citations(review(quote=" "), row(), [])

    def test_answer_cannot_cite_after_action_results(self):
        retrieved = [{"id": "later", "text": "The answer appears after the action."}]
        with self.assertRaisesRegex(ValueError, "answer_citation_after_action"):
            validate_citations(review("retrieved_after_search", retrieved[0]["text"], "later"),
                                row("answer"), retrieved)

    def test_search_result_and_prior_context_bind_to_their_own_documents(self):
        source = {"proposed_action": {"action": "search"},
                  "before_action": [{"id": "prior", "text": "Known: 1995."}]}
        citations = [
            {"stage": "before_action", "document_id": "prior", "quote": "Known: 1995."},
            {"stage": "retrieved_after_search", "document_id": "later", "quote": "Missing fact: 1997."},
        ]
        self.assertEqual(validate_citations({"citations": citations}, source,
                                            [{"id": "later", "text": "Missing fact: 1997."}]), 2)

    def test_rejects_quote_spliced_across_documents(self):
        source = {"proposed_action": {"action": "answer"}, "before_action": [
            {"id": "a", "text": "The year was 1995."},
            {"id": "b", "text": "The band was The Get Up Kids."},
        ]}
        v = {"citations": [{"stage": "before_action", "document_id": "a",
                            "quote": "The year was 1995. The band was The Get Up Kids."}]}
        with self.assertRaisesRegex(ValueError, "citation_not_exactly_bound_to_one_document"):
            validate_citations(v, source, [])


if __name__ == "__main__":
    unittest.main()
