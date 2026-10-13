"""Ensure the candidate Luna review packet is blind and no-gold."""

import unittest

from scripts.v128_package_embedded_search_luna import audit


class EmbeddedSearchLunaPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit()

    def test_packet_contains_only_six_anonymized_search_states(self):
        packet = self.result["packet"]
        self.assertEqual(len(packet["cases"]), 6)
        self.assertEqual(len({case["case_id"] for case in packet["cases"]}), 6)
        self.assertTrue(all(case["proposed_action"]["action"] == "search"
                            for case in packet["cases"]))
        self.assertTrue(all(set(case) == {
            "case_id", "question", "before_action", "proposed_action", "retrieved_after_search"
        } for case in packet["cases"]))
        self.assertTrue(all(doc["id"].startswith("doc-")
                            for case in packet["cases"]
                            for stage in ("before_action", "retrieved_after_search")
                            for doc in case[stage]))
        self.assertTrue(all(doc["title"].strip() and doc["text"].strip()
                            for case in packet["cases"]
                            for stage in ("before_action", "retrieved_after_search")
                            for doc in case[stage]))
        self.assertGreater(self.result["manifest"][
            "retrieved_prior_document_references_resolved"], 0)
        for case in packet["cases"]:
            before_by_id = {doc["id"]: doc for doc in case["before_action"]}
            for doc in case["retrieved_after_search"]:
                if doc.get("resolved_prior_reference"):
                    self.assertEqual(doc["text"], before_by_id[doc["id"]]["text"])

    def test_no_source_ids_answer_targets_or_reviewer_votes_are_disclosed(self):
        encoded = self.result["packet_bytes"].decode("utf-8")
        for private_marker in ("v117_train:", "candidate_id_private", "answer_action",
                               "source_answer_record_id", "approved_training_label",
                               "gold_answer", "semantic_double_review"):
            self.assertNotIn(private_marker, encoded)
        manifest = self.result["manifest"]
        self.assertFalse(manifest["gold_fields_in_packet"])
        self.assertFalse(manifest["source_ids_in_packet"])
        self.assertFalse(manifest["answer_targets_in_packet"])
        self.assertFalse(manifest["other_reviewer_votes_in_packet"])
        self.assertFalse(manifest["training_allowed"])


if __name__ == "__main__":
    unittest.main()
