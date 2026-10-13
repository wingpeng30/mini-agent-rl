"""Regression tests for rejection-vote evidence binding in the prospective v129 protocol."""
import copy
import json
import unittest
from pathlib import Path

from scripts.v128_review_protocol import valid_vote as valid_v128
from scripts.v129_review_protocol import contract_identity, valid_vote as valid_v129

ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "data/evidence-sft-v124-forensic/goal-v128"


class ReviewProtocolV129Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = json.loads((GOAL / "g2-embedded-search-luna-packet-private.json").read_text(encoding="utf-8"))
        cls.votes = json.loads((GOAL / "g2-embedded-search-luna-response-private.json").read_text(encoding="utf-8"))

    def test_old_case04_vote_shows_v128_rejection_citation_gap(self):
        packet = next(x for x in self.packet["cases"] if x["case_id"] == "case-04")
        vote = next(x for x in self.votes if x["case_id"] == "case-04")
        self.assertTrue(valid_v128(packet, vote)[0])
        self.assertEqual(valid_v129(packet, vote),
                         (False, "search_vote_missing_action_time_prior_citation"))

    def test_rejection_requires_action_time_and_retrieval_citations(self):
        packet = next(x for x in self.packet["cases"] if x["case_id"] == "case-04")
        vote = copy.deepcopy(next(x for x in self.votes if x["case_id"] == "case-04"))
        vote["citations"].append({
            "stage": "before_action", "step": None, "document_id": "doc-02",
            "quote": "It was introduced in the show and on the RCA Victor cast album by Jack Cassidy.",
        })
        self.assertTrue(valid_v129(packet, vote)[0])
        vote["citations"] = [c for c in vote["citations"] if c["stage"] != "retrieved_after_search"]
        self.assertEqual(valid_v129(packet, vote),
                         (False, "search_vote_missing_retrieval_citation"))

    def test_contract_is_distinct_from_immutable_v128(self):
        identity = contract_identity()
        self.assertEqual(identity["version"], "v129-evidence-review-v1")
        self.assertNotEqual(identity["system_sha256"], self.packet["protocol"]["system_sha256"])
        manifest = json.loads((GOAL / "review-contract-v2-private.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["protocol"], identity)
        self.assertEqual(manifest["submitted_packet_count"], 0)


if __name__ == "__main__":
    unittest.main()
