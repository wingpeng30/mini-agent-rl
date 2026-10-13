"""Real-tokenizer regression checks for Goal G's CPU acceptance gate."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from transformers import AutoProcessor

from scripts.run_v133_goal_g_cpu import classify_offsets, cpu_check, make_case

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/root/autodl-tmp/qwen_08b")


@pytest.fixture(scope="module")
def fixture_case():
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    path = ROOT / "runs/v133-goal-g-cpu/review-candidates.jsonl"
    item = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return processor, make_case(item)


def test_normal_prefix_and_token_groups(fixture_case):
    processor, case = fixture_case
    result = cpu_check(processor, case)
    assert result["answer_tokens"] > 0
    assert result["target_tokens"] == result["answer_tokens"] + result["format_tokens"]


def test_wrong_target(fixture_case):
    processor, case = fixture_case
    bad = copy.deepcopy(case)
    bad["target_assistant_index"] = 5
    with pytest.raises(ValueError, match="wrong_target_assistant"):
        cpu_check(processor, bad)


def test_missing_evidence(fixture_case):
    processor, case = fixture_case
    bad = copy.deepcopy(case)
    bad["messages"][5]["content"] = '{"ok":true,"result":[]}'
    with pytest.raises(ValueError, match="missing_one_of_two_evidence_documents"):
        cpu_check(processor, bad)


def test_overlong_evidence(fixture_case):
    processor, case = fixture_case
    bad = copy.deepcopy(case)
    doc = bad["evidence_documents"][1]
    doc["text"] += " repeated" * 5000
    bad["messages"][5]["content"] = json.dumps({"ok": True, "result": [doc]}, ensure_ascii=False, separators=(",", ":"))
    with pytest.raises(ValueError, match="max_length"):
        cpu_check(processor, bad)


def test_target_message_leakage(fixture_case):
    processor, case = fixture_case
    bad = copy.deepcopy(case)
    bad["messages"][2]["content"] = bad["messages"][6]["content"]
    with pytest.raises(ValueError, match="target_message_leaked"):
        cpu_check(processor, bad)


def test_json_escape_or_explicit_boundary(fixture_case):
    processor, case = fixture_case
    changed = copy.deepcopy(case)
    changed["official_answer"] = 'a "quoted" category'
    changed["messages"][6]["content"] = json.dumps({"action": "answer", "answer": changed["official_answer"]}, ensure_ascii=False)
    try:
        result = cpu_check(processor, changed)
        assert result["answer_tokens"] > 0
    except ValueError as exc:
        assert "answer_format_token_boundary_ambiguous" in str(exc)


def test_cross_boundary_is_detected():
    assert classify_offsets([(0, 6), (6, 10), (10, 14)], 5, 10) == (1, 1, 1)
