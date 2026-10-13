"""Goal Z source isolation and actual two-step retrieval invariants."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import prepare_v133_goal_z_multihop_data as z
import run_v133_goal_z_multihop_sft as runner

OUT = ROOT / "runs/v133-goal-z-multihop-sft"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_fixed_pool_counts_and_source_archives():
    report = json.loads((OUT / "source-audit.json").read_text(encoding="utf-8"))
    curation = json.loads((OUT / "curation.json").read_text(encoding="utf-8"))
    assert curation["shortfalls"] == {}
    assert curation["counts"] == {f"{split}-{source}": quota
                                  for split, quotas in z.TARGETS.items()
                                  for source, quota in quotas.items()}
    assert report["source_archives"] == {"musique": z.digest(z.MUSIQUE), "2wiki": z.digest(z.WIKI)}
    for name, info in report["pools"].items():
        split, source = name.split("-")
        assert info["candidates"] == z.LIMITS[split][source]
        assert info["file_sha256"] == z.digest(OUT / f"candidates-{split}-{source}.jsonl")


def test_split_isolation_and_corpus_size():
    selected = {split: rows(OUT / ("selected-confirm.jsonl" if split == "confirm"
                                     else f"selected-{split}.jsonl"))
                for split in ("train", "dev", "confirm")}
    corpus_titles = {}
    for split, values in selected.items():
        assert len(values) == {"train": 128, "dev": 64, "confirm": 64}[split]
        corpus = json.loads((OUT / f"corpus-{split}.json").read_text(encoding="utf-8"))
        assert len(corpus) == len({z.norm(d["title"]) for d in corpus}) == 2048
        corpus_titles[split] = {z.norm(d["title"]) for d in corpus}
        for case in values:
            if case["source"] != "old":
                assert {z.norm(x) for x in case.get("support_titles", case.get("teacher", {}).get("support_titles", []))}.issubset(corpus_titles[split])
    for a, b in (("train", "dev"), ("train", "confirm"), ("dev", "confirm")):
        assert not corpus_titles[a] & corpus_titles[b]
        assert not {x["id"] for x in selected[a]} & {x["id"] for x in selected[b]}
        assert not {z.norm(x["question"]) for x in selected[a]} & {z.norm(x["question"]) for x in selected[b]}
        assert not {v for x in selected[a] for v in x["source_ids"]} & {
            v for x in selected[b] for v in x["source_ids"]}


def test_training_trajectories_need_second_search_and_visible_bridge():
    train = rows(OUT / "selected-train.jsonl")
    assert sum(x["source"] == "old" for x in train) == 32
    for case in train:
        if case["source"] == "old":
            assert case["answer_prompt_token_ids"]
            assert case["answer_prompt_attention_mask"]
            continue
        teacher = case["teacher"]
        first, second = teacher["retrieved"]
        assert len(teacher["queries"]) == 2
        assert len(first) <= 3 and len(second) <= 3
        assert z.norm(teacher["bridge"]) not in z.norm(case["question"])
        assert z.norm(teacher["bridge"]) not in z.norm(teacher["queries"][0])
        assert teacher["bridge_visible_in_doc_ids"]
        assert set(teacher["bridge_visible_in_doc_ids"]).issubset({d["id"] for d in first})
        assert z.norm(teacher["support_titles"][0]) in {z.norm(d["title"]) for d in first}
        assert z.norm(teacher["support_titles"][1]) not in {z.norm(d["title"]) for d in first}
        assert z.norm(teacher["support_titles"][1]) in {z.norm(d["title"]) for d in second}


def test_materialized_actions_and_matching_seed_orders():
    inputs = rows(OUT / "training-inputs.jsonl")
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    assert len(inputs) == 128
    assert set(config["order"]["42"]) == set(config["order"]["43"]) == {x["id"] for x in inputs}
    assert len(config["order"]["42"]) == len(config["order"]["43"]) == 128
    for case in inputs:
        actions = case["actions"]
        if case["source"] == "old":
            assert len(actions) in (2, 3)
        else:
            assert len(actions) == 3
        assert actions[-1]["action"] == "answer"
        assert all(action["action"] == "search" for action in actions[:-1])
        assert len(actions) <= 3
        for action in actions:
            assert action["action_token_ids"]
            assert len(action["prompt_token_ids"]) == len(action["prompt_attention_mask"]) <= 2048
            assert len(action["action_token_ids"]) <= 128
    assert sum(len(x["actions"]) == 3 for x in inputs) == 98


def test_four_runs_complete_and_fixed_inputs_replay_exactly():
    baseline = {x["case_id"]: x for x in rows(OUT / "eval-dev-incumbent.jsonl")}
    assert len(baseline) == 64
    for label in ("42-answer", "43-answer", "42-trajectory", "43-trajectory"):
        folder = OUT / f"seed{label}"
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        updates = rows(folder / "updates.jsonl")
        assert len(updates) == summary["updates"] == 128
        assert [x["step"] for x in updates] == list(range(1, 129))
        assert summary["base_unchanged"] and summary["changed_lora_tensors"] > 0
        assert summary["reload_max_abs"] <= .001
        assert runner.sha(folder / "adapter-final/adapter_model.safetensors") == summary["adapter_sha256"]
    for label in ("incumbent", "42-answer", "43-answer", "42-trajectory", "43-trajectory"):
        fixed = rows(OUT / f"fixed-dev-{label}.jsonl")
        assert len(fixed) == 64
        for row in fixed:
            prompt = baseline[row["case_id"]]["trace"][-1]["prompt_token_ids"]
            expected = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            assert row["prompt_sha256"] == expected


def test_failed_development_gate_keeps_confirmation_closed():
    decision = json.loads((OUT / "development-decision.json").read_text(encoding="utf-8"))
    state = json.loads((OUT / "state.json").read_text(encoding="utf-8"))
    assert decision["status"] == state["status"] == "complete_development_gate_failed"
    assert decision["selected_route"] is None
    assert not list(OUT.glob("eval-confirm-*.jsonl"))
    try:
        runner.confirm("incumbent")
    except RuntimeError as exc:
        assert "Confirmation gate closed" in str(exc)
    else:
        raise AssertionError("Confirmation was accessible before its gate")
