"""Goal W's data isolation and pretraining stop invariants."""
from __future__ import annotations

import json
from pathlib import Path

import run_v133_goal_w_transfer as w


def test_frozen_split_titles_and_questions_do_not_overlap():
    data = w.DATA
    manifest = w.verify_data_manifest()
    assert manifest["historical_exclusions"]["reconstructed_goal_v"]["rows"] == 300
    cases = {split: w.rows(data / f"candidate-{split}.jsonl")
             for split in ("train", "dev", "confirm")}
    titles = {split: {doc["title"].casefold() for doc in
                      json.loads((data / f"corpus-{split}.json").read_text(encoding="utf-8"))["documents"]}
              for split in cases}
    for left, right in (("train", "dev"), ("train", "confirm"), ("dev", "confirm")):
        assert not titles[left] & titles[right]
        assert not {x["source_id"] for x in cases[left]} & {x["source_id"] for x in cases[right]}
        assert not {x["question"].casefold() for x in cases[left]} & \
                   {x["question"].casefold() for x in cases[right]}


def test_goal_stopped_before_new_model_work():
    result = w.summarize()
    assert result["baseline"] == {"total": 20, "regression": 7, "invalid": 0}
    assert result["status"] == "complete_data_shortfall"
    assert result["training_started"] is False
    assert result["question_audit"] == {
        "different_asked_attribute": 16,
        "possibly_shared_group_relation": 12,
    }
    assert result["decision"]["generous_upper_bound"] < result["decision"]["minimum_required"]
    assert not any(Path(w.OUT).glob("seed*-*/adapter-final/adapter_model.safetensors"))
