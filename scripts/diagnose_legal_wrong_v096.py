"""Read-only counterfactual audit for the v096 strategy-reward pilot.

The script deliberately does not select training tasks.  It scores every
stored v095 corrected rollout with one pre-registered candidate reward and
reports whether that reward has enough applicable legal-but-wrong cases to
justify a separate pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from mini_agent_rl.domain import Task, Transition
from mini_agent_rl.reward.functions import (
    CompositeReward, EmptyEvidenceAnswerPenalty, EvidenceReward,
    EvidenceStopReward, ExactMatchReward, IncompleteEvidenceAnswerPenalty,
    InvalidActionPenalty, MaxStepsPenalty, OverSearchAfterEvidencePenalty,
    RepeatedSearchPenalty, SearchCostPenalty, SupportCoverageReward,
    UngroundedAnswerPenalty,
)
from mini_agent_rl.storage.sqlite import SQLiteStore


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_reward() -> CompositeReward:
    return CompositeReward([
        ExactMatchReward(1.0), EvidenceReward(), SupportCoverageReward(),
        SearchCostPenalty(), RepeatedSearchPenalty(-0.5), InvalidActionPenalty(),
        MaxStepsPenalty(), UngroundedAnswerPenalty(), OverSearchAfterEvidencePenalty(),
        EvidenceStopReward(0.20), IncompleteEvidenceAnswerPenalty(-0.15),
        EmptyEvidenceAnswerPenalty(-0.10),
    ])


def analyze(db_path: Path) -> dict:
    store = SQLiteStore(db_path)
    by_group: dict[str, list[dict]] = defaultdict(list)
    changed_wrong = changed_empty = full_correct_changed = 0
    for rollout in store.list_rollouts():
        task_row = store.conn.execute("SELECT data FROM tasks WHERE id=?", (rollout.task_id,)).fetchone()
        task = Task.model_validate(json.loads(task_row[0])) if task_row else None
        transitions = store.get_transitions(rollout.id)
        if task is None:
            raise RuntimeError(f"missing task: {rollout.id}")
        parts = candidate_reward().evaluate(task, transitions, rollout.final_answer)
        total = sum(parts.values())
        legal_wrong = rollout.termination_reason == "answer" and parts["exact_match"] == 0.0
        changed_wrong += int(legal_wrong and abs(total - rollout.reward) > 1e-9)
        changed_empty += int(legal_wrong and parts["empty_evidence_answer"] != 0.0)
        full_correct_changed += int(parts["exact_match"] > 0 and parts["evidence_stop"] > 0 and abs(total - rollout.reward) > 1e-9)
        by_group[rollout.group_id].append({"old": float(rollout.reward), "new": total, "legal_wrong": legal_wrong,
                                           "exact": parts["exact_match"], "parts": parts})
    group_rows = []
    for group_id, rows in by_group.items():
        old = [row["old"] for row in rows]; new = [row["new"] for row in rows]
        group_rows.append({"group_id": group_id, "old_variance": len(set(old)) > 1,
                           "new_variance": len(set(new)) > 1,
                           "changed": any(abs(a-b) > 1e-9 for a, b in zip(old, new)),
                           "correct_beats_legal_wrong": any(a["exact"] > 0 and b["legal_wrong"] and a["new"] > b["new"]
                                                              for a in rows for b in rows)})
    store.close()
    return {
        "db": str(db_path), "sha256": sha(db_path), "groups": len(group_rows),
        "legal_wrong_reward_changed": changed_wrong,
        "empty_result_wrong_reward_changed": changed_empty,
        "fully_supported_correct_reward_changed": full_correct_changed,
        "old_nonzero_variance_groups": sum(row["old_variance"] for row in group_rows),
        "new_nonzero_variance_groups": sum(row["new_variance"] for row in group_rows),
        "changed_groups": sum(row["changed"] for row in group_rows),
        "groups_correct_beats_legal_wrong": sum(row["correct_beats_legal_wrong"] for row in group_rows),
        "details": group_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [analyze(path) for path in args.db]
    affected = sum(item["legal_wrong_reward_changed"] for item in reports)
    changed_groups = sum(item["changed_groups"] for item in reports)
    # Frozen before the diagnostic: at least ten applicable legal-wrong
    # rollouts and six affected groups across the three existing seeds.
    proceed = len(reports) == 3 and affected >= 10 and changed_groups >= 6
    result = {
        "version": "v0.9.6", "kind": "legal_wrong_counterfactual_reward_diagnostic",
        "candidate": {"evidence_stop_weight": 0.20, "incomplete_evidence_answer_penalty": -0.15,
                      "empty_evidence_answer_penalty": -0.10},
        "pre_registered_gate": {"seed_count": 3, "minimum_affected_legal_wrong_rollouts": 10,
                                 "minimum_changed_groups": 6, "passed": proceed},
        "seeds": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
