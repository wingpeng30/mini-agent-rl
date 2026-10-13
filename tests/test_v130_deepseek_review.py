from __future__ import annotations

import json
import os
import shutil
import uuid
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import v130_deepseek_review as review
from scripts.v130_deepseek_review import ROOT, checked_packet_path, run_batch


def packet():
    return {"case_id": "anon-1", "question": "Where was the artist born?",
            "visible_messages": [{"role": "user", "content": "Where was the artist born?"}],
            "before_action": [{"id": "doc-1", "title": "Artist", "text": "The artist was born in Oslo."}],
            "proposed_action": {"action": "answer", "answer": "Oslo"},
            "retrieved_after_search": []}


def vote(*, approved=True, citations=True):
    return {"case_id": "anon-1", "approved": approved,
            "reason": "The visible source identifies the birth place.",
            "risk_labels": [], "missing_fact": "",
            "search_needed": "no", "query_grounded": "not_available",
            "retrieval_relevant": "not_available",
            "citations": ([{"stage": "before_action", "step": None,
                            "document_id": "doc-1", "quote": "The artist was born in Oslo."}]
                          if citations else [])}


def response(decision=None, *, finish_reason="stop", content=None):
    message_content = json.dumps(decision) if content is None else content
    return json.dumps({"choices": [{"finish_reason": finish_reason,
                                     "message": {"content": message_content}}]}).encode()


class FakeResponse:
    status = 200
    def __init__(self, body):
        self.body = body
    def read(self):
        return self.body


class FakeOpener:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
    def open(self, request, timeout):
        self.calls.append((request, timeout))
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


class DeepSeekReviewV130Tests(unittest.TestCase):
    def run_temp(self, opener, batch="batch-01"):
        path = ROOT / f".v130-review-test-{uuid.uuid4().hex}-{batch}"
        self.addCleanup(shutil.rmtree, path, True)
        result = run_batch(batch=batch, packets=[packet()], packet_file_sha256="a" * 64,
                           freeze_sha256="b" * 64, batch_dir=path,
                           key="test-secret-never-log", opener=opener,
                           clock=lambda: 1234567890)
        return path, result

    def test_valid_vote_is_persisted_without_credential_and_batch_cannot_repeat(self):
        opener = FakeOpener([response({"reviews": [vote()]})])
        path, result = self.run_temp(opener)
        self.assertEqual(result["reviews"], [vote()])
        self.assertEqual(len(opener.calls), 1)
        manifest = json.loads((path / "manifest-private.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["freeze_sha256"], "b" * 64)
        self.assertEqual(manifest["protocol"]["version"], "v130-evidence-review-citations-v1")
        self.assertEqual(manifest["request_profile"]["thinking"], {"type": "disabled"})
        self.assertEqual(manifest["request_profile"]["max_tokens"], 8192)
        self.assertEqual(manifest["request_profile"]["transport"], "configured_proxy")
        self.assertEqual(manifest["packet_count"], 1)
        body = json.loads(opener.calls[0][0].data)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["max_tokens"], 8192)
        for name in ("manifest-private.json", "request-journal-private.jsonl",
                     "validated-review-private.json"):
            text = (path / name).read_text(encoding="utf-8")
            self.assertNotIn("test-secret-never-log", text)
        with self.assertRaisesRegex(RuntimeError, "batch_exists"):
            run_batch(batch="batch-01", packets=[packet()], packet_file_sha256="a" * 64,
                      freeze_sha256="b" * 64, batch_dir=path,
                      key="test-secret-never-log", opener=opener)
        with self.assertRaisesRegex(RuntimeError, "identical_payload_previously_attempted"):
            run_batch(batch="batch-02", packets=[packet()], packet_file_sha256="a" * 64,
                      freeze_sha256="b" * 64,
                      batch_dir=ROOT / f"{path.name}-second",
                      key="test-secret-never-log",
                      opener=opener)
        self.assertEqual(len(opener.calls), 1)

    def test_unknown_network_result_stops_and_same_batch_cannot_retry(self):
        opener = FakeOpener([TimeoutError("simulated timeout")])
        path = ROOT / f".v130-review-test-{uuid.uuid4().hex}-unknown"
        self.addCleanup(shutil.rmtree, path, True)
        with self.assertRaisesRegex(RuntimeError, "outcome_unknown"):
            run_batch(batch="batch-01", packets=[packet()], packet_file_sha256="a" * 64,
                      freeze_sha256="b" * 64, batch_dir=path, key="key", opener=opener)
        journal = (path / "request-journal-private.jsonl").read_text(encoding="utf-8")
        self.assertIn("outcome_unknown_stop_without_retry", journal)
        self.assertIn('"error_type":"TimeoutError"', journal)
        with self.assertRaisesRegex(RuntimeError, "batch_exists"):
            run_batch(batch="batch-01", packets=[packet()], packet_file_sha256="a" * 64,
                      freeze_sha256="b" * 64, batch_dir=path, key="key", opener=opener)
        self.assertEqual(len(opener.calls), 1)

    def test_only_one_technical_correction_is_allowed(self):
        invalid = {"reviews": [vote(citations=False)]}
        opener = FakeOpener([response(invalid), response({"reviews": [vote()]})])
        path, result = self.run_temp(opener)
        self.assertEqual(len(opener.calls), 2)
        second_body = json.loads(opener.calls[1][0].data)
        self.assertIn("technical_correction", second_body["messages"][1]["content"])
        journal = (path / "request-journal-private.jsonl").read_text(encoding="utf-8")
        self.assertIn("technical_validation_failed_one_correction_allowed", journal)
        self.assertEqual(result["reviews"], [vote()])

    def test_length_limited_empty_completion_stops_without_technical_retry(self):
        opener = FakeOpener([response(finish_reason="length", content="")])
        path = ROOT / f".v130-review-test-{uuid.uuid4().hex}-length"
        self.addCleanup(shutil.rmtree, path, True)
        with self.assertRaisesRegex(RuntimeError, "empty_or_truncated_completion"):
            run_batch(batch="batch-01", packets=[packet()], packet_file_sha256="a" * 64,
                      freeze_sha256="b" * 64, batch_dir=path,
                      key="key", opener=opener)
        journal = (path / "request-journal-private.jsonl").read_text(encoding="utf-8")
        self.assertIn("empty_or_truncated_completion_stop_without_retry", journal)
        self.assertNotIn("technical_validation_failed_one_correction_allowed", journal)
        self.assertEqual(len(opener.calls), 1)

    def test_semantic_rejection_is_saved_without_correction(self):
        rejected = vote(approved=False, citations=False)
        opener = FakeOpener([response({"reviews": [rejected]})])
        _, result = self.run_temp(opener)
        self.assertFalse(result["reviews"][0]["approved"])
        self.assertEqual(len(opener.calls), 1)

    def test_packet_paths_are_rejected_before_read_outside_allowlist_or_sealed_scope(self):
        with self.assertRaisesRegex(ValueError, "sealed_packet_path_rejected_before_read"):
            checked_packet_path("data/hotpot-agent-v072/final-test.jsonl")
        with self.assertRaisesRegex(ValueError, "packet_path_rejected_before_read"):
            checked_packet_path("data/evidence-sft-v124-forensic/goal-v128/new-internal-audit/packets/../x.json")
        with self.assertRaisesRegex(ValueError, "packet_path_outside_new_internal_audit_allowlist"):
            checked_packet_path("data/evidence-sft-v124-forensic/inventory-private.jsonl")

    def test_process_environment_credentials_are_supported_without_creating_env_file(self):
        absent = ROOT / "this-path-must-not-exist-v130.env"
        with patch.object(review, "ENV_FILE", absent), patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "private-test-key", "DEEPSEEK_API_BASE": "https://api.deepseek.com/"
        }, clear=True):
            key, base = review._credentials()
        self.assertEqual(key, "private-test-key")
        self.assertEqual(base, "https://api.deepseek.com")
        self.assertFalse(absent.exists())

    def test_direct_opener_explicitly_disables_ambient_proxy(self):
        with patch.object(review.urllib.request, "build_opener") as build:
            review._no_redirect_opener(direct_connection=True)
        handlers = build.call_args.args
        proxy_handlers = [h for h in handlers if isinstance(h, review.urllib.request.ProxyHandler)]
        self.assertEqual(len(proxy_handlers), 1)
        self.assertEqual(proxy_handlers[0].proxies, {})


if __name__ == "__main__":
    unittest.main()
