import hashlib
import json

import pytest

from scripts.check_action_protocol_contract import check_probe, check_sft
from scripts.diagnose_v121_protocol import classify


def test_sft_targets_use_the_runtime_search_answer_protocol(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"messages": [{"role": "assistant", "content": '{"action":"search","query":"Atlanta population"}'}], "target_assistant_index": 0},
        {"messages": [{"role": "assistant", "content": '{"action":"answer","answer":"Atlanta"}'}], "target_assistant_index": 0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert check_sft(path) == {"rows": 2, "search": 1, "answer": 1}
    rows[1]["messages"][0]["content"] = '{"action":"answer","answer":{"status":"complete"}}'
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="target_protocol_invalid"):
        check_sft(path)


def test_probe_binds_every_generation_to_frozen_settings_and_blocks_invalid_output(tmp_path):
    path = tmp_path / "probe.jsonl"
    settings_hash = hashlib.sha256(b"frozen settings").hexdigest()
    good = {"settings_sha256": settings_hash, "raw_output": '{"action":"search","query":"birth date"}<|im_end|>', "output_truncated": False}
    path.write_text(json.dumps(good), encoding="utf-8")
    assert check_probe(path, settings_hash)["rate"] == 1
    with pytest.raises(ValueError, match="settings_hash_mismatch"):
        check_probe(path, "another hash")
    bad = {**good, "raw_output": '{"action":"answer","answer":"{"status":"complete"}"}'}
    path.write_text("\n".join(json.dumps(row) for row in (good, bad)), encoding="utf-8")
    with pytest.raises(ValueError, match="probe_protocol_gate_failed"):
        check_probe(path, settings_hash)


def test_diagnosis_separates_old_action_from_nested_json_failure():
    assert classify({"raw_output": '{"action":"search","query":"birth date"}<|im_end|>'}) == "old_search_action"
    assert classify({"raw_output": '{"action":"answer","answer":"{"status":"complete"}"}'}) == "unescaped_nested_json"
