"""CPU-only gates; synthetic inputs only, with no sealed data access."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.v123_cpu_core import safe, validate_archive_members, validate_selected_pairs, validate_training
from scripts.v123_query_gate import query_check
from scripts.v123_api_protocol import validate as validate_review
from scripts.v123_verify_upload import verify_extract
from scripts.v123_prepare_internal import packet_same


def test_sealed_paths_and_archive_members_rejected(tmp_path):
    for name in ("final-test", "final_test", "independent100", "holdout"):
        with pytest.raises(ValueError, match="sealed_path_forbidden"):
            safe(tmp_path / name / "sample.json", expect=tmp_path)
        archive = tmp_path / "candidate-archive.zip"
        with ZipFile(archive, "w") as stream:
            stream.writestr("safe/" + name + "/payload.json", "{}");
        with pytest.raises(ValueError, match="forbidden_archive_member"):
            validate_archive_members(archive, expect=tmp_path)
    archive = tmp_path / "traversal.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("../unsafe.json", "{}")
    with pytest.raises(ValueError, match="forbidden_archive_member"):
        validate_archive_members(archive, expect=tmp_path)


def test_lexical_overlap_is_diagnostic_only():
    values = [
        ("3OH!3 band", "What label signed 3OH!3?", True),
        ("new york county", "Which New York city is involved?", True),
        ("alias punctuation", "What is the alias?", True),
        ("unseen named place", "Where did the event occur?", False),
    ]
    for query, question, _ in values:
        result = query_check(query, set(), question, [])
        assert result["allow"] is True and result["semantic_grounding"] == "unverified"
    assert query_check("", set(), "Q", [])["reason"] == "empty_query"
    assert query_check("YORK", {"york"}, "Q", [])["reason"] == "repeated_query"


def _packet(qid, action):
    return {"case_id": qid + action, "question": qid,
            "proposed_action": {"action": action}, "visible_messages": [{"role": "user", "content": qid}]}


def test_internal_exact_pair_and_risk_gate():
    packets = [_packet(str(i), kind) for i in range(50) for kind in ("search", "answer")]
    source = {p["case_id"]: p["question"] for p in packets}
    risk = {p["case_id"]: ["relation", "time", "measurement"] + (["scope"] if int(p["question"]) < 2 else [])
            for p in packets}
    report = validate_selected_pairs(packets, source, [str(i) for i in range(50)], risk, set(), set(), set())
    assert report["passed"] is False and report["risks"]["scope"] == 4
    fewer = validate_selected_pairs(packets[:-2], source, [str(i) for i in range(49)], risk, set(), set(), set())
    assert fewer["checks"]["fifty_distinct_questions"] is False
    risk[packets[4]["case_id"]].append("scope")
    assert validate_selected_pairs(packets, source, [str(i) for i in range(50)], risk, set(), set(), set())["passed"]


def _training_row(question, action):
    target = {"action": action, "query": question} if action == "search" else {"action": action, "answer": question}
    messages = [{"role": "user", "content": question}]
    if action == "search":
        messages.append({"role": "tool", "content": json.dumps({"result": [{"id": "d", "text": "fact"}]})})
    target_index = len(messages)
    messages.append({"role": "assistant", "content": json.dumps(target)})
    return {"messages": messages, "target_assistant_index": target_index}


def test_training_counts_do_not_imply_permission():
    base = [_training_row(f"q{i}", "search" if n == 0 else "answer")
            for i in range(75) for n in (0, 1)]
    base += [_training_row(f"q{i}", "search") for i in range(75, 270)]
    supplement = [_training_row(f"q{i}", kind) for i in range(270, 295) for kind in ("search", "answer")]
    report = validate_training(base, supplement, [f"q{i}" for i in range(270, 295)], set(), set(), set(), set())
    assert report["checks"]["395_states"] and report["checks"]["100_complete_pairs"]
    assert report["checks"]["211_unique_questions"] is False
    assert report["passed"] is False and report["training_allowed"] is False
    changed = [dict(row) for row in supplement]
    changed[0] = {**changed[0], "messages": [{"role": "user", "content": changed[0]["messages"][0]["content"]},
                                              changed[0]["messages"][-1]], "target_assistant_index": 1}
    fewer = validate_training(base, changed, [f"q{i}" for i in range(270, 295)], set(), set(), set(), set())
    assert fewer["checks"]["395_states"] and not fewer["checks"]["100_complete_pairs"]


def test_old_vote_reuse_requires_exact_visible_input_and_token_hash():
    old = {"question": "Q", "visible_messages": [{"role": "user", "content": "Q"}],
           "before_action": [{"id": "1", "text": "A"}],
           "proposed_action": {"action": "search", "query": "A"},
           "retrieved_after_search": [{"id": "2", "text": "B"}, {"id": "3", "text": "C"}],
           "prompt_tokens": 20, "target_tokens": 5, "total_tokens": 25,
           "prompt_ids_sha256": "a", "sequence_ids_sha256": "b", "target_mask_sha256": "c"}
    assert packet_same(old, dict(old))
    assert not packet_same(old, {**old, "retrieved_after_search": list(reversed(old["retrieved_after_search"]))})
    assert not packet_same(old, {**old, "prompt_ids_sha256": "changed"})


def test_review_quote_is_bound_to_one_document_and_action_time():
    packet = {"case_id": "c", "proposed_action": {"action": "answer"},
              "before_action": [{"id": "one", "text": "Alpha is in Paris."},
                                {"id": "two", "text": "Beta was born in 1900."}],
              "retrieved_after_search": [{"id": "three", "text": "Alpha was born in 1900."}]}
    base = {"case_id": "c", "approved": True, "reason": "supported", "risk_labels": ["relation"],
            "citations": [{"step": None, "stage": "before_action", "document_id": "one",
                           "quote": "Alpha is in Paris."}]}
    validate_review(packet, base)
    with pytest.raises(ValueError, match="case_id_mismatch"):
        validate_review(packet, {**base, "case_id": "wrong"})
    with pytest.raises(ValueError, match="approved_not_boolean"):
        validate_review(packet, {**base, "approved": "yes"})
    for citation in (
        {"step": None, "stage": "before_action", "document_id": "one", "quote": "Alpha is in Paris. Beta"},
        {"step": None, "stage": "before_action", "document_id": "three", "quote": "Alpha was born in 1900."},
        {"step": None, "stage": "retrieved_after_search", "document_id": "three", "quote": "Alpha was born in 1900."},
    ):
        with pytest.raises(ValueError):
            validate_review(packet, {**base, "citations": [citation]})


def test_transfer_verifies_before_extracting(tmp_path):
    archive = tmp_path / "safe.zip"
    data = b"private synthetic packet"
    import hashlib
    digest = hashlib.sha256(data).hexdigest()
    manifest = {"version": "v123-transfer-v1", "files": {"data/p.json": digest},
                "training_allowed": False, "credentials_included": False, "sealed_assets_included": False}
    with ZipFile(archive, "w") as stream:
        stream.writestr("data/p.json", data)
        stream.writestr("v123-transfer-manifest.json", json.dumps(manifest))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="transport_hash"):
        verify_extract(archive, output, "0" * 64)
    assert not output.exists()
    result = verify_extract(archive, output, hashlib.sha256(archive.read_bytes()).hexdigest())
    assert result["status"] == "passed" and (output / "data/p.json").read_bytes() == data
