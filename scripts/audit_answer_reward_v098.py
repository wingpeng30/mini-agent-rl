"""Read-only v098 answer-reward audit and frozen-sampling gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from mini_agent_rl.domain import Task, Transition
from mini_agent_rl.reward.functions import (
    AnswerTokenF1Reward, CompositeReward, EmptyEvidenceAnswerPenalty,
    EvidenceReward, EvidenceStopReward, ExactMatchReward, IncompleteEvidenceAnswerPenalty,
    InvalidActionPenalty, MaxStepsPenalty, OverSearchAfterEvidencePenalty,
    RepeatedSearchPenalty, SearchCostPenalty, SupportCoverageReward,
    UngroundedAnswerPenalty,
)
from mini_agent_rl.answer_scoring import normalize_answer, reward_token_f1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate() -> CompositeReward:
    return CompositeReward([
        ExactMatchReward(1.0), AnswerTokenF1Reward(0.20), SupportCoverageReward(),
        SearchCostPenalty(), RepeatedSearchPenalty(-0.5), InvalidActionPenalty(),
        MaxStepsPenalty(), UngroundedAnswerPenalty(), OverSearchAfterEvidencePenalty(),
        EvidenceStopReward(0.20), IncompleteEvidenceAnswerPenalty(-0.15),
        EmptyEvidenceAnswerPenalty(-0.10),
    ], version="v098-answer-v1")


def baseline() -> CompositeReward:
    return CompositeReward([
        ExactMatchReward(1.0), EvidenceReward(), SupportCoverageReward(),
        SearchCostPenalty(), RepeatedSearchPenalty(-0.5), InvalidActionPenalty(),
        MaxStepsPenalty(), UngroundedAnswerPenalty(), OverSearchAfterEvidencePenalty(),
        EvidenceStopReward(0.20), IncompleteEvidenceAnswerPenalty(-0.15),
        EmptyEvidenceAnswerPenalty(-0.10),
    ], version="v096")


def read(path: Path) -> dict:
    before = sha(path)
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    rows = [json.loads(row[0]) for row in conn.execute("SELECT data FROM rollouts")]
    transitions = defaultdict(list)
    for row in conn.execute("SELECT data FROM transitions ORDER BY step"):
        item = json.loads(row[0]); transitions[item["rollout_id"]].append(item)
    tasks = {row[0]: json.loads(row[1]) for row in conn.execute("SELECT id,data FROM tasks")}
    conn.close()
    result = []
    scorer = candidate()
    old_scorer = baseline()
    for rollout in rows:
        task = Task.model_validate(tasks[rollout["task_id"]])
        own = [Transition.model_validate(item) for item in transitions[rollout["id"]]]
        parts = scorer.evaluate(task, own, rollout.get("final_answer"))
        total = sum(parts.values())
        old_parts = old_scorer.evaluate(task, own, rollout.get("final_answer"))
        old_total = sum(old_parts.values())
        final = rollout.get("final_answer") or ""
        result.append({
            "task_id": rollout["task_id"], "group_id": rollout["group_id"], "rollout_id": rollout["id"],
            "termination": rollout.get("termination_reason"),
            "stored_reward": float(rollout.get("reward", 0.0)),
            "old_reward": old_total,
            "old_parts": old_parts,
            "candidate_reward": total, "candidate_parts": parts,
            # Keep the benchmark's literal EM separate from the more forgiving
            # normalization used by F1.  The latter is useful diagnostic context,
            # but must not turn aliases or formatting variants into an EM win.
            "strict_exact": float(final.strip().lower() == task.answer.strip().lower()),
            "normalized_exact": float(normalize_answer(final) == normalize_answer(task.answer)),
            "token_f1": reward_token_f1(final, task.answer), "answer": final, "gold": task.answer,
            "support_titles": task.metadata.get("support_titles", []),
            "visible_message_hash": hashlib.sha256(json.dumps([m.model_dump(mode="json") for t in own for m in t.messages], sort_keys=True).encode()).hexdigest(),
        })
    after = sha(path)
    if before != after:
        raise RuntimeError(f"source changed during read-only audit: {path}")
    return {"db": str(path), "sha256_before": before, "sha256_after": after, "rows": result}


def summarize(runs: list[dict]) -> dict:
    rows = [row for run in runs for row in run["rows"]]
    groups = defaultdict(list)
    by_task = defaultdict(list)
    for row in rows:
        groups[(row["db"] if "db" in row else "", row["group_id"])].append(row)
        by_task[row["task_id"]].append(row)
    distinguish = 0; inversions = 0
    f1_varying = 0
    advantage_changed = 0
    improved_correct_margins = 0
    worsened_correct_margins = 0
    for values in groups.values():
        legal = [v for v in values if v["termination"] == "answer"]
        if len({(v["strict_exact"], v["token_f1"], v["candidate_reward"]) for v in legal}) > 1:
            distinguish += 1
        if len({v["token_f1"] for v in legal}) > 1:
            f1_varying += 1
        old_mean = sum(v["old_reward"] for v in values) / len(values)
        candidate_mean = sum(v["candidate_reward"] for v in values) / len(values)
        if any(abs((v["candidate_reward"] - candidate_mean) - (v["old_reward"] - old_mean)) > 1e-12 for v in values):
            advantage_changed += 1
        inversions += sum(wrong["candidate_reward"] > correct["candidate_reward"] for wrong in legal for correct in legal if not wrong["strict_exact"] and correct["strict_exact"])
        for wrong in legal:
            for correct in legal:
                if wrong["strict_exact"] or not correct["strict_exact"]:
                    continue
                old_margin = correct["old_reward"] - wrong["old_reward"]
                candidate_margin = correct["candidate_reward"] - wrong["candidate_reward"]
                improved_correct_margins += candidate_margin > old_margin + 1e-12
                worsened_correct_margins += candidate_margin < old_margin - 1e-12
    review_ids = {"hotpot-5abd90545542996e802b47d7", "hotpot-5ab381b155429969a97a816b", "hotpot-5adc1309554299438c868d3b"}
    review = [row for row in rows if row["task_id"] in review_ids]
    candidate_nonzero = sum(len({v["candidate_reward"] for v in values}) > 1 for values in groups.values())
    old_nonzero = sum(len({v["old_reward"] for v in values}) > 1 for values in groups.values())
    candidate_only = sum(len({v["candidate_reward"] for v in values}) > 1 and len({v["old_reward"] for v in values}) == 1 for values in groups.values())
    return {"rollouts": len(rows), "group_count": len(groups),
            "old_nonzero_variance_groups": old_nonzero,
            "candidate_nonzero_variance_groups": candidate_nonzero,
            "candidate_only_nonzero_groups": candidate_only,
            "legal_token_f1_varying_groups": f1_varying,
            "centered_advantage_changed_groups": advantage_changed,
            "improved_correct_wrong_pair_margins": improved_correct_margins,
            "worsened_correct_wrong_pair_margins": worsened_correct_margins,
            "legal_rank_distinguishing_groups": distinguish, "correct_below_wrong_pairs": inversions,
            "review_cases": review}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-gate", action="store_true")
    args = parser.parse_args()
    runs = [read(path) for path in args.db]
    for run in runs:
        for row in run["rows"]: row["db"] = run["db"]
    summary = summarize(runs)
    gate = None
    if args.frozen_gate:
        per_seed = []
        for run in runs:
            groups = defaultdict(list)
            for row in run["rows"]: groups[row["group_id"]].append(row)
            per_seed.append(sum(len({row["candidate_reward"] for row in values}) > 1 for values in groups.values()))
        gate = {"nonzero_groups_per_seed": per_seed, "minimum_per_seed": 5,
                "minimum_distinguishing_groups": 3, "inversions": summary["correct_below_wrong_pairs"],
                "passed": len(runs) == 3 and all(value >= 5 for value in per_seed) and summary["legal_rank_distinguishing_groups"] >= 3 and summary["correct_below_wrong_pairs"] == 0}
    payload = {"version": "v0.9.8", "reward_version": "v098-answer-v1", "sources": [{k: v for k, v in run.items() if k != "rows"} for run in runs], "summary": summary, "frozen_gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
