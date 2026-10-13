"""Goal Y's new material and matching protocol invariants."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v133_goal_x_audit_transfer as x

OUT = ROOT / "runs/v133-goal-y-three-class-transfer"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def test_pairs_use_actual_saved_tokens_and_audited_three_classes():
    frozen = read(OUT / "pairs.json")["pairs"]
    trace = {v["case_id"]: v for v in lines(x.OUT / "train-real-search.jsonl")}
    assert len(frozen) == len({v["case_id"] for v in frozen}) == 25
    assert Counter(v["category"] for v in frozen) == {"occupation": 12, "entity_type": 8, "origin": 5}
    for pair in frozen:
        original = trace[pair["case_id"]]
        assert original["both_support_full_text_visible"]
        assert pair["input_ids"] == original["answer_prompt_token_ids"]
        assert pair["input_mask"] == original["answer_prompt_attention_mask"]
        assert pair["chosen"] == original["official_answer"]
        assert pair["chosen"] != pair["rejected"]


def test_clean_dev_is_disjoint_from_historical_titles():
    cases = lines(OUT / "dev-cases.jsonl")
    docs = read(OUT / "dev-corpus.json")
    titles = set(read(x.OUT / "x0-history-exposure-index.json")["post_g_title_union"]["normalized_titles"])
    titles |= x.history_index()[2]
    assert len(cases) == 48
    assert Counter(c["attribute_category"] for c in cases) == {
        "occupation": 16, "entity_type": 16, "source": 16}
    assert not {x.norm(d["title"]) for d in docs} & titles
    assert {v for c in cases for v in c["support_ids"]}.issubset({d["id"] for d in docs})
    proposals = lines(OUT / "pair-proposals.jsonl")
    train_titles = {x.norm(d["title"]) for item in proposals
                    for d in item["evidence"]["chosen_both_sides"]}
    assert not train_titles & {x.norm(d["title"]) for d in docs}
    assert not {item["case_id"] for item in proposals} & {case["case_id"] for case in cases}


def test_remote_frozen_schedules_match_both_arms_and_hard_cap():
    cfg = read(OUT / "config.json")
    ids = {v["case_id"] for v in read(OUT / "pairs.json")["pairs"]}
    assert cfg["arms"] == ["a", "b"]
    assert cfg["seeds"] == [42, 43]
    assert cfg["updates_per_run"] == 25
    for seed in ("42", "43"):
        assert len(cfg["order"][seed]) == 25
        assert set(cfg["order"][seed]) == ids
    assert cfg["gpu_time_limit"] is None and cfg["stage_timeouts"] is None
