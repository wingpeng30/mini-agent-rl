"""Corrected clustered statistics and full v101 error taxonomy (read-only)."""
from __future__ import annotations

import argparse, hashlib, json, random, re, sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from mini_agent_rl.answer_scoring import normalize_answer, token_f1

SEEDS = (42, 43, 44)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path: Path, name: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return [json.loads(row[0]) for row in conn.execute(f"SELECT data FROM {name}")]
    finally:
        conn.close()


def load(path: Path) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, dict]]:
    rollouts = {row["task_id"]: row for row in table(path, "rollouts")}
    transitions: dict[str, list[dict]] = defaultdict(list)
    for row in table(path, "transitions"):
        transitions[row["rollout_id"]].append(row)
    for values in transitions.values(): values.sort(key=lambda row: row["step"])
    tasks = {row["id"]: row for row in table(path, "tasks")}
    return rollouts, transitions, tasks


def run_audit(path: Path) -> dict:
    groups=table(path,"rollout_groups")
    environments={json.dumps(row.get("environment_fingerprint",{}),sort_keys=True) for row in groups}
    system_prompts=set()
    for transition in table(path,"transitions"):
        for message in transition.get("messages",[]):
            if message.get("role")=="system": system_prompts.add(message.get("content",""))
    return {"environment_fingerprints":sorted(environments),"system_prompt_sha256":sorted(hashlib.sha256(x.encode()).hexdigest() for x in system_prompts),"group_count":len(groups)}


def tool_docs_from_actual_messages(transitions: list[dict]) -> list[dict]:
    if not transitions: return []
    final = transitions[-1]
    docs = []
    for message in final.get("messages", []):
        if message.get("role") != "tool": continue
        try: payload = json.loads(message.get("content", ""))
        except (TypeError, json.JSONDecodeError): continue
        docs.extend(item for item in (payload.get("result") or []) if isinstance(item, dict))
    return docs


def classify(rollout: dict, transitions: list[dict], task: dict) -> dict:
    gold = task.get("answer", ""); answer = rollout.get("final_answer") or ""
    gold_n, answer_n = normalize_answer(gold), normalize_answer(answer)
    strict_gold = re.sub(r"\s+", " ", gold.strip().casefold())
    strict_answer = re.sub(r"\s+", " ", answer.strip().casefold())
    f1 = token_f1(answer, gold); exact = strict_gold == strict_answer
    normalized_exact = gold_n == answer_n
    docs = tool_docs_from_actual_messages(transitions)
    visible = " ".join(f"{d.get('title','')} {d.get('text','')}" for d in docs)
    visible_n = normalize_answer(visible)
    titles = {str(d.get("title", "")).casefold() for d in docs}
    wanted = {str(x).casefold() for x in task.get("metadata", {}).get("support_titles", [])}
    support_complete = bool(wanted) and wanted <= titles
    gold_visible = bool(gold_n and gold_n in visible_n)
    reason = rollout.get("termination_reason")
    if exact: category = "correct"
    elif reason != "answer": category = "protocol_or_termination"
    elif normalized_exact or f1 >= .8 or (gold_n and answer_n and (gold_n in answer_n or answer_n in gold_n)):
        category = "answer_variant_candidate"
    elif not docs: category = "ungrounded_direct_answer"
    elif gold_visible: category = "entity_or_relation_selection_error"
    elif support_complete: category = "support_complete_but_inference_error"
    elif wanted & titles: category = "partial_support_then_answer"
    else: category = "retrieval_failure_or_wrong_query"
    return {"task_id": rollout["task_id"], "rollout_id": rollout["id"], "question": task.get("question", ""), "answer": answer, "gold": gold,
            "exact": exact, "normalized_exact": normalized_exact, "token_f1": f1, "termination_reason": reason, "category": category,
            "gold_string_in_actual_prompt": gold_visible, "support_titles_in_actual_prompt": sorted(wanted & titles),
            "support_titles_complete": support_complete, "visible_titles": sorted(titles),
            "visible_evidence_excerpt": visible[:1200],
            "actual_prompt_sha256": hashlib.sha256(json.dumps(transitions[-1].get("messages", []), sort_keys=True, ensure_ascii=False).encode()).hexdigest() if transitions else None}


def paired_bootstrap(by_task: dict[str, float], repeats: int = 10_000) -> dict:
    values = list(by_task.values()); rng = random.Random(42); n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(repeats))
    return {"n_tasks": n, "delta": sum(values)/n, "low": draws[int(.025*repeats)], "high": draws[int(.975*repeats)]}


def reward_gate(root: Path) -> dict:
    per_seed={}; total_changed=total_inversions=0; affected=set(); correct_affected=set()
    for seed in SEEDS:
        path=root/"runs"/f"v100-answer30-seed{seed}.db"; before=sha(path)
        rollouts=table(path,"rollouts"); transitions=table(path,"transitions"); tasks={x["id"]:x for x in table(path,"tasks")}
        by_rollout=defaultdict(list)
        for row in transitions: by_rollout[row["rollout_id"]].append(row)
        by_group=defaultdict(list)
        for rollout in rollouts:
            tagged=sum("over_search_after_evidence" in t.get("reward_tags",[]) for t in by_rollout[rollout["id"]])
            old=float(rollout.get("reward",0)); new=old+.2*tagged
            task=tasks[rollout["task_id"]]
            correct=re.sub(r"\s+"," ",(rollout.get("final_answer") or "").strip().casefold())==re.sub(r"\s+"," ",(task.get("answer") or "").strip().casefold())
            if tagged:
                affected.add(rollout["task_id"])
                if correct: correct_affected.add(rollout["task_id"])
            by_group[rollout["group_id"]].append({"old":old,"new":new,"correct":correct})
        changed=inversions=variance=0
        for values in by_group.values():
            old=[x["old"] for x in values]; new=[x["new"] for x in values]
            old_center=[x-sum(old)/len(old) for x in old]; new_center=[x-sum(new)/len(new) for x in new]
            changed += int(any(abs(a-b)>1e-9 for a,b in zip(old_center,new_center)))
            variance += int(len({round(x,12) for x in new})>1)
            inversions += sum(a["new"] < b["new"] for a in values if a["correct"] for b in values if not b["correct"])
        total_changed+=changed; total_inversions+=inversions
        per_seed[str(seed)]={"changed_centered_advantage_groups":changed,"nonzero_variance_groups":variance,"correct_below_wrong_pairs":inversions,"source_hash_unchanged":before==sha(path)}
    passed=len(affected)>=3 and len(correct_affected)>=3 and total_changed>=6 and total_inversions==0 and all(x["nonzero_variance_groups"]>=15 for x in per_seed.values())
    return {"affected_tasks":len(affected),"correct_affected_tasks":len(correct_affected),"changed_centered_advantage_groups":total_changed,"correct_below_wrong_pairs":total_inversions,"per_seed":per_seed,"gate_passed":passed}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--v099-root",type=Path,required=True); ap.add_argument("--v100-root",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    paths = {"sft": args.v099_root/"runs"/"v099-eval100-sft.db"}
    for version in ("v100","v101"):
        for seed in SEEDS: paths[f"{version}-{seed}"]=args.v100_root/"runs"/f"{version}-eval100-seed{seed}.db"
    before={name:sha(path) for name,path in paths.items()}; loaded={name:load(path) for name,path in paths.items()}; audits={name:run_audit(path) for name,path in paths.items()}
    all_rows=[]; metrics={}; task_ids=sorted(loaded["sft"][0])
    for name,(rollouts,transitions,tasks) in loaded.items():
        model_rows={task_id:classify(rollouts[task_id],transitions[rollouts[task_id]["id"]],tasks[task_id]) for task_id in task_ids}
        metrics[name]={"em":sum(r["exact"] for r in model_rows.values())/len(task_ids),"f1":sum(r["token_f1"] for r in model_rows.values())/len(task_ids),"errors":sum(not r["exact"] for r in model_rows.values())}
        if name.startswith("v101"):
            seed=int(name.split("-")[1])
            for row in model_rows.values(): all_rows.append({"seed":seed,**row})
    comparisons={}
    for baseline in ("sft","v100"):
        for metric in ("exact","token_f1"):
            diffs={}
            for task_id in task_ids:
                candidate=sum(classify(loaded[f"v101-{s}"][0][task_id],loaded[f"v101-{s}"][1][loaded[f"v101-{s}"][0][task_id]["id"]],loaded[f"v101-{s}"][2][task_id])[metric] for s in SEEDS)/3
                if baseline=="sft":
                    base=classify(loaded["sft"][0][task_id],loaded["sft"][1][loaded["sft"][0][task_id]["id"]],loaded["sft"][2][task_id])[metric]
                else:
                    base=sum(classify(loaded[f"v100-{s}"][0][task_id],loaded[f"v100-{s}"][1][loaded[f"v100-{s}"][0][task_id]["id"]],loaded[f"v100-{s}"][2][task_id])[metric] for s in SEEDS)/3
                diffs[task_id]=float(candidate)-float(base)
            comparisons[f"v101_minus_{baseline}_{metric}"]=paired_bootstrap(diffs)
    errors=[row for row in all_rows if not row["exact"]]
    per_task=defaultdict(Counter)
    for row in errors: per_task[row["task_id"]][row["category"]]+=1
    taxonomy={"rollout_counts":dict(Counter(r["category"] for r in errors)),"unique_error_tasks":len(per_task),
              "tasks_wrong_all_three_seeds":sum(sum(c.values())==3 for c in per_task.values()),
              "per_task_category_counts":{k:dict(v) for k,v in per_task.items()}}
    after={name:sha(path) for name,path in paths.items()}
    comparable_env=len({json.dumps(x["environment_fingerprints"],sort_keys=True) for x in audits.values()})==1
    comparable_prompt=len({json.dumps(x["system_prompt_sha256"]) for x in audits.values()})==1
    report={"version":"v103","cluster_unit":"100 validation questions","bootstrap_seed":42,"bootstrap_repeats":10000,
            "source_hashes_unchanged":before==after,"comparable_environment":comparable_env,"comparable_system_prompt":comparable_prompt,
            "run_audits":audits,"metrics":metrics,"paired":comparisons,"taxonomy":taxonomy,
            "corrected_oversearch_reward_gate":reward_gate(args.v100_root),
            "limitations":["answer variants are candidates for review, not automatically accepted","support-title completeness does not prove the answer is entailed"]}
    (args.output_dir/"v103-corrected-analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (args.output_dir/"v103-error-details.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in errors)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
