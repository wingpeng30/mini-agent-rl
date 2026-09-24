"""Read-only paired decomposition of v094/v095 GRPO evaluation artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.prompts import DIRECT_SYSTEM
from mini_agent_rl.storage.sqlite import SQLiteStore


SEEDS = (42, 43, 44)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(deltas: list[float], seed: int = 42, repeats: int = 10_000) -> dict:
    if not deltas:
        return {"n": 0, "delta": None, "low": None, "high": None}
    rng = random.Random(seed)
    draws = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(repeats))
    return {"n": len(deltas), "delta": sum(deltas) / len(deltas),
            "low": draws[int(repeats * .025)], "high": draws[min(repeats - 1, int(repeats * .975))]}


def load_tasks(path: Path) -> dict[str, dict]:
    if "final-test" in str(path).lower() or "v072" in str(path).lower():
        raise ValueError("禁止读取封存最终集")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["metadata"]["task_id"]): {"answer": row["metadata"]["gold_answer"],
            "support_titles": row["metadata"].get("support_titles", [])} for row in rows}


def coverage(titles: set[str], wanted: set[str]) -> str:
    if not wanted or not titles:
        return "none"
    return "complete" if wanted <= titles else "partial"


def rollout_row(store: SQLiteStore, rollout, task: dict) -> dict:
    transitions = store.get_transitions(rollout.id)
    wanted = {str(item).casefold() for item in task["support_titles"]}
    found: set[str] = set()
    partial_continue = False
    oversearch_complete = False
    empty_seen = False
    system_prompts = set()
    for transition in transitions:
        system_prompts.update(message.content for message in transition.messages if message.role == "system")
        call = transition.response.tool_call
        before = coverage(found, wanted)
        if call and call.name == "search":
            partial_continue |= before == "partial"
            oversearch_complete |= before == "complete"
            result = (transition.observation or {}).get("result") or []
            empty_seen |= not result
            found.update(str(doc.get("title", "")).casefold() for doc in result if isinstance(doc, dict))
    metric = answer_metrics(rollout.final_answer or "", task["answer"])
    invalid = rollout.termination_reason == "invalid_action"
    answered = rollout.termination_reason == "answer"
    legal = not invalid and rollout.termination_reason not in {"model_error", "tool_error", "cancelled"}
    final_coverage = coverage(found, wanted)
    return {"task_id": rollout.task_id, "rollout_id": rollout.id, "termination_reason": rollout.termination_reason,
            "legal_action": legal, "invalid_action": invalid, "answered": answered,
            "strict_exact_match": metric["strict_exact_match"], "token_f1": metric["token_f1"],
            "repeated_action": rollout.termination_reason == "repeated_action",
            "max_steps": rollout.termination_reason == "max_steps",
            "output_budget_exhaustion": any(bool(t.response.usage.get("output_truncated")) for t in transitions),
            "empty_result_direct_answer": bool(answered and empty_seen),
            "empty_result_direct_answer_wrong": bool(answered and empty_seen and not metric["strict_exact_match"]),
            "partial_coverage_continue_search": partial_continue,
            "complete_coverage_answer_correct": bool(answered and final_coverage == "complete" and metric["strict_exact_match"]),
            "oversearch_after_complete": oversearch_complete,
            "final_coverage": final_coverage, "system_prompt_hashes": sorted(hashlib.sha256(item.encode()).hexdigest() for item in system_prompts),
            "system_prompt_exact": system_prompts == {DIRECT_SYSTEM}}


def load_run(path: Path, tasks: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    store = SQLiteStore(path)
    rows = {}
    environments = set()
    for rollout in store.list_rollouts():
        if rollout.task_id not in tasks:
            continue
        row = rollout_row(store, rollout, tasks[rollout.task_id])
        if rollout.task_id in rows:
            raise RuntimeError(f"{path} has repeated task {rollout.task_id}")
        rows[rollout.task_id] = row
        group = store.get_group(rollout.group_id)
        if group:
            environments.add(json.dumps(group.environment_fingerprint, sort_keys=True))
    store.close()
    audit = {"db_sha256": sha(path), "task_count": len(rows), "task_ids_sha256": hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest(),
             "all_direct_system": all(row["system_prompt_exact"] for row in rows.values()),
             "environment_fingerprints": sorted(environments)}
    return rows, audit


def probe_audit(root: Path, prefix: str) -> dict:
    expected = None
    reports = {}
    for name in ("sft", *(f"seed{seed}" for seed in SEEDS)):
        cases = root / f"{prefix}-{name}" / "cases.jsonl"
        rows = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines() if line.strip()]
        normalized = {row["case_id"]: [message for message in row["messages"] if message["role"] != "system"] for row in rows}
        current = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        reports[name] = {"cases_sha256": sha(cases), "case_count": len(rows),
                         "all_direct_system": all([message["content"] for message in row["messages"] if message["role"] == "system"] == [DIRECT_SYSTEM] for row in rows),
                         "non_system_messages_sha256": hashlib.sha256(current.encode()).hexdigest()}
        expected = expected or reports[name]["non_system_messages_sha256"]
    return {"runs": reports, "non_system_history_identical": all(item["non_system_messages_sha256"] == expected for item in reports.values())}


def probe_behavior(root: Path, prefix: str) -> dict:
    """Compare frozen probe choices by original question, never by 120 pseudo-cases."""
    rows_by_name = {}
    for name in ("sft", *(f"seed{seed}" for seed in SEEDS)):
        rows = [json.loads(line) for line in (root / f"{prefix}-{name}" / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        grouped = {}
        for row in rows:
            grouped.setdefault(row["task_id"], {})[row["condition"]] = row
        if len(grouped) != 30 or any(len(item) != 4 for item in grouped.values()):
            raise RuntimeError(f"探针分组不完整：{name}")
        rows_by_name[name] = grouped
    result = {}
    for seed in SEEDS:
        name = f"seed{seed}"; by_condition = {}
        for condition in ("one_hop", "two_hop", "empty_result", "repeated_history"):
            deltas = {key: [] for key in ("answered", "strict_exact_match", "token_f1", "search", "repeated_query", "invalid_action", "wrong_direct_answer")}
            for task_id, base_cases in rows_by_name["sft"].items():
                left, right = base_cases[condition], rows_by_name[name][task_id][condition]
                left_values = {"answered": left["answered"], "strict_exact_match": left["strict_exact_match"], "token_f1": left["token_f1"],
                               "search": left["search"], "repeated_query": left["repeated_query"], "invalid_action": left["invalid_action"],
                               "wrong_direct_answer": left["answered"] and not left["strict_exact_match"]}
                right_values = {"answered": right["answered"], "strict_exact_match": right["strict_exact_match"], "token_f1": right["token_f1"],
                                "search": right["search"], "repeated_query": right["repeated_query"], "invalid_action": right["invalid_action"],
                                "wrong_direct_answer": right["answered"] and not right["strict_exact_match"]}
                for key in deltas:
                    deltas[key].append(float(right_values[key]) - float(left_values[key]))
            by_condition[condition] = {key: bootstrap(values) for key, values in deltas.items()}
        result[str(seed)] = by_condition
    return result


def movement(base: dict, candidate: dict) -> str:
    if base["invalid_action"] and candidate["legal_action"]:
        return "invalid_to_legal_correct" if candidate["strict_exact_match"] else "invalid_to_legal_wrong"
    if base["legal_action"] and candidate["legal_action"]:
        if not base["strict_exact_match"] and candidate["strict_exact_match"]:
            return "legal_wrong_to_correct"
        if base["strict_exact_match"] and not candidate["strict_exact_match"]:
            return "legal_correct_to_wrong"
        return "legal_stable_correct" if base["strict_exact_match"] else "legal_stable_wrong"
    return "other"


def analyze(base: dict[str, dict], candidates: dict[int, dict[str, dict]]) -> tuple[dict, list[dict]]:
    ids = sorted(base)
    if any(sorted(rows) != ids for rows in candidates.values()):
        raise RuntimeError("评测任务集合不一致")
    details, per_seed = [], {}
    for seed, rows in candidates.items():
        moves = {}
        pairs = []
        for task_id in ids:
            left, right = base[task_id], rows[task_id]
            label = movement(left, right)
            moves[label] = moves.get(label, 0) + 1
            detail = {"task_id": task_id, "seed": seed, "movement": label, "sft": left, "grpo": right}
            details.append(detail)
            if left["legal_action"] and right["legal_action"]:
                pairs.append((left, right))
        per_seed[str(seed)] = {
            "movement_counts": moves,
            "protocol_recovery": {"em_contribution": sum(right["strict_exact_match"] for left, right in zip((base[i] for i in ids), (rows[i] for i in ids)) if left["invalid_action"] and right["legal_action"]) / len(ids),
                                  "f1_contribution": sum(right["token_f1"] for left, right in zip((base[i] for i in ids), (rows[i] for i in ids)) if left["invalid_action"] and right["legal_action"]) / len(ids)},
            "legal_pair_em": bootstrap([right["strict_exact_match"] - left["strict_exact_match"] for left, right in pairs]),
            "legal_pair_f1": bootstrap([right["token_f1"] - left["token_f1"] for left, right in pairs]),
            "strategy_deltas": {key: bootstrap([float(rows[i][key]) - float(base[i][key]) for i in ids]) for key in
                                ("partial_coverage_continue_search", "complete_coverage_answer_correct", "oversearch_after_complete",
                                 "empty_result_direct_answer", "empty_result_direct_answer_wrong", "repeated_action", "max_steps", "output_budget_exhaustion")}}
    common = [task_id for task_id in ids if base[task_id]["legal_action"] and all(rows[task_id]["legal_action"] for rows in candidates.values())]
    cross = {
        "common_legal_task_count": len(common),
        "legal_pair_em": bootstrap([sum(rows[i]["strict_exact_match"] for rows in candidates.values()) / len(candidates) - base[i]["strict_exact_match"] for i in common]),
        "legal_pair_f1": bootstrap([sum(rows[i]["token_f1"] for rows in candidates.values()) / len(candidates) - base[i]["token_f1"] for i in common]),
        "strategy_deltas": {key: bootstrap([sum(float(rows[i][key]) for rows in candidates.values()) / len(candidates) - float(base[i][key]) for i in ids]) for key in
                            ("partial_coverage_continue_search", "complete_coverage_answer_correct", "oversearch_after_complete",
                             "empty_result_direct_answer", "empty_result_direct_answer_wrong", "repeated_action", "max_steps", "output_budget_exhaustion")}}
    return {"per_seed": per_seed, "cross_seed": cross}, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/hotpot-agent-v041/test.jsonl")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--probe-root", default="reports")
    parser.add_argument("--run-prefix", default="v094-full100", help="Prefix before -sft.db / -seed42.db")
    parser.add_argument("--probe-prefix", default="v094-probe120", help="Prefix before -sft/ -seed42 probe directories")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data, runs, root = Path(args.data), Path(args.runs_dir), Path(args.probe_root)
    tasks = load_tasks(data)
    base, base_audit = load_run(runs / f"{args.run_prefix}-sft.db", tasks)
    candidates, audits = {}, {"sft": base_audit}
    for seed in SEEDS:
        name = f"seed{seed}"
        candidates[seed], audits[name] = load_run(runs / f"{args.run_prefix}-{name}.db", tasks)
    decomposition, details = analyze(base, candidates)
    probe = probe_audit(root, args.probe_prefix)
    probe["behavior_deltas"] = probe_behavior(root, args.probe_prefix)
    protocol_ok = all(item["all_direct_system"] for item in audits.values()) and probe["non_system_history_identical"] and all(item["all_direct_system"] for item in probe["runs"].values())
    legal_em = decomposition["cross_seed"]["legal_pair_em"]
    legal_f1 = decomposition["cross_seed"]["legal_pair_f1"]
    if legal_em["low"] is not None and legal_em["low"] > 0 and legal_f1["low"] > 0:
        conclusion = "协议与合法轨迹问答均有统计支持的改善；可在P0通过后研究50题扩量。"
    elif legal_em["delta"] is not None and legal_em["delta"] > 0:
        conclusion = "合法轨迹存在改善迹象，但区间未完全排除零；先保持30题结论，不扩量。"
    else:
        conclusion = "总体改善主要来自协议恢复或合法轨迹未改善；先诊断reward和策略。"
    report = {"version": "v0.9.5", "data_sha256": sha(data), "protocol_audit_passed": protocol_ok,
              "run_audit": audits, "probe_audit": probe, "decomposition": decomposition, "conclusion": conclusion}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in details) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(
        "# v095：GRPO效果来源分解\n\n"
        f"- 协议审计通过：{protocol_ok}\n"
        f"- 共同合法轨迹：{decomposition['cross_seed']['common_legal_task_count']} 题\n"
        f"- 合法轨迹 EM 差：{legal_em['delta']}，95% CI [{legal_em['low']}, {legal_em['high']}]\n"
        f"- 合法轨迹 F1 差：{legal_f1['delta']}，95% CI [{legal_f1['low']}, {legal_f1['high']}]\n\n"
        f"结论：{conclusion}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
