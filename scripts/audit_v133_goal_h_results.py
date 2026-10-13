"""Bind Goal H's changed correct outputs to Goal G's exact-context dual reviews."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "runs/v133-goal-h-comparison"
G = ROOT / "runs/v133-goal-g-cpu"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


summary = json.loads((H / "summary.json").read_text(encoding="utf-8"))
assert summary["status"] == "complete_exploratory_synthetic_history_comparison"
assert sha(H / "trajectories.jsonl") == summary["trajectories_sha256"]
assert sha(G / "freeze-manifest.json") == "76605092f8582111093138ba14c31204b3303ffe8025d87a3dce2dd3fa661c36"
packets = {r["case_id"]: r for r in rows(G / "review-packets.jsonl")}
evaluation = {r["case_id"]: r for r in rows(G / "evaluation-frozen.jsonl")}
trajectories = {(r["arm"], r["case_id"]): r for r in rows(H / "trajectories.jsonl") if r["split"] == "main"}
luna = {r["case_id"]: r for r in json.loads((G / "luna-votes-technical-correction.json").read_text(encoding="utf-8"))}
disposition = {r["case_id"]: r for r in rows(G / "vote-disposition.jsonl")}
out = []
for case_id in sorted(set(summary["changes_vs_v118"]["sft_d"]["new_correct"]) |
                      set(summary["changes_vs_v118"]["sft_c"]["new_correct"])):
    packet = packets[case_id]
    case = evaluation[case_id]
    assert packet["visible_messages"] == case["messages"][:case["target_assistant_index"]]
    assert packet["question"] == case["messages"][1]["content"]
    assert packet["proposed_action"] == json.loads(case["messages"][-1]["content"])
    assert packet["before_action"] == case["evidence_documents"]
    batch = case["review_refs"]["deepseek_batch"]
    deepseek_path = G / "deepseek" / batch / "validated-review-private.json"
    deepseek_obj = json.loads(deepseek_path.read_text(encoding="utf-8"))
    deepseek = next(r for r in deepseek_obj["reviews"] if r["case_id"] == case_id)
    assert deepseek["approved"] and luna[case_id]["approved"]
    assert disposition[case_id]["disposition"] == "qualified"
    arms = []
    for arm in ("sft_c", "sft_d"):
        trajectory = trajectories[(arm, case_id)]
        if trajectory["em"]:
            assert json.loads(trajectory["raw_output"]) == packet["proposed_action"]
            assert trajectory["prompt_sha256"] == case["cpu_check"]["prompt_sha256"]
            arms.append(arm)
    out.append({"case_id": case_id, "matching_arms": arms,
                "exact_action_and_prompt_match": True, "qualified_in_G": True,
                "deepseek_approved": True, "luna_approved": True,
                "deepseek_vote_sha256": sha(deepseek_path),
                "luna_votes_sha256": sha(G / "luna-votes-technical-correction.json"),
                "evidence_titles": case["evidence_titles"],
                "review_reuse_scope": "same answer action and identical visible messages; no new review call"})

audit = {"version": "v133-goal-h-review-binding-v1", "cases": out,
         "source_packet_sha256": sha(G / "review-packets.jsonl"),
         "source_disposition_sha256": sha(G / "vote-disposition.jsonl"),
         "source_eval_sha256": sha(G / "evaluation-frozen.jsonl"),
         "new_review_calls": 0}
(H / "review-binding.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"review_binding_pass": True, "cases": [r["case_id"] for r in out]}, ensure_ascii=False))
