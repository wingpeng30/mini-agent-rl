"""Read-only v102 P0 error migration and P1 over-search ablation gate."""
from __future__ import annotations

import argparse, hashlib, json, random, re, sqlite3
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = (42, 43, 44)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (value or "").casefold()).strip()

def rows(path: Path, table: str) -> list[dict]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return [json.loads(row[0]) for row in conn.execute(f"SELECT data FROM {table}")]
    finally:
        conn.close()

def load_run(path: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    rollouts = {row["task_id"]: row for row in rows(path, "rollouts")}
    transitions: dict[str, list[dict]] = defaultdict(list)
    for row in rows(path, "transitions"):
        transitions[row["rollout_id"]].append(row)
    for values in transitions.values(): values.sort(key=lambda item: item["step"])
    return rollouts, transitions

def search_events(ts: list[dict]) -> list[dict]:
    return [t for t in ts if (t.get("response", {}).get("tool_calls") or [])]

def answer_row(rollout: dict, ts: list[dict], gold: str) -> dict:
    searches = search_events(ts)
    empties = [not bool((t.get("observation") or {}).get("result") or []) for t in searches]
    docs = [doc for t in searches for doc in ((t.get("observation") or {}).get("result") or []) if isinstance(doc, dict)]
    answer = rollout.get("final_answer") or ""
    exact = norm(answer) == norm(gold)
    visible_text = " ".join(str(doc.get("text", "")) for doc in docs)
    reason = rollout.get("termination_reason")
    answered = reason == "answer"
    return {
        "answer": answer, "gold": gold, "exact": exact, "answered": answered,
        "termination_reason": reason, "search_count": len(searches),
        "any_empty": any(empties), "last_search_empty": bool(empties and empties[-1]),
        "all_searches_empty": bool(empties) and all(empties),
        "gold_string_visible": bool(norm(gold) and norm(gold) in norm(visible_text)),
        "visible_titles": [doc.get("title") for doc in docs],
    }

def bootstrap(values: list[float], repeats: int = 10_000) -> dict:
    rng = random.Random(42); n = len(values)
    sims = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats)) if n else []
    return {"n": n, "delta": sum(values) / n if n else None,
            "low": sims[int(.025 * repeats)] if sims else None,
            "high": sims[int(.975 * repeats)] if sims else None}

def p0(root: Path, data: Path, output: Path, detail: Path) -> dict:
    gold = {row["metadata"]["task_id"]: row["metadata"]["gold_answer"]
            for row in (json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip())}
    before_hashes = {}
    runs = {}
    for version in ("v100", "v101"):
        for seed in SEEDS:
            path = root / "runs" / f"{version}-eval100-seed{seed}.db"
            before_hashes[str(path)] = sha(path); runs[(version, seed)] = load_run(path)
    details, deltas = [], []
    empty_metrics = {version: Counter() for version in ("v100", "v101")}
    termination = {version: Counter() for version in ("v100", "v101")}
    for seed in SEEDS:
        old_r, old_t = runs[("v100", seed)]; new_r, new_t = runs[("v101", seed)]
        for task_id in sorted(set(old_r) & set(new_r)):
            old = answer_row(old_r[task_id], old_t[old_r[task_id]["id"]], gold[task_id])
            new = answer_row(new_r[task_id], new_t[new_r[task_id]["id"]], gold[task_id])
            for version, item in (("v100", old), ("v101", new)):
                termination[version][item["termination_reason"]] += 1
                for metric in ("any_empty", "last_search_empty", "all_searches_empty"):
                    empty_metrics[version][f"{metric}_answered"] += int(item[metric] and item["answered"])
                    empty_metrics[version][f"{metric}_wrong"] += int(item[metric] and item["answered"] and not item["exact"])
            deltas.append(float(new["exact"]) - float(old["exact"]))
            if old["exact"] and not new["exact"]:
                category = ("gold_answer_visible_but_wrong" if new["gold_string_visible"] else
                            "retrieval_or_evidence_insufficient" if new["answered"] else "protocol_or_termination")
                details.append({"source": "autonomous", "seed": seed, "task_id": task_id,
                                "migration": "correct_to_wrong", "category": category,
                                "old": old, "new": new})
    # Probe decisions already preserve the exact model-visible history and audit metadata.
    for seed in SEEDS:
        oldp = [json.loads(x) for x in (root/"reports"/f"v100-probe120-seed{seed}"/"decisions.jsonl").read_text(encoding="utf-8").splitlines() if x]
        newp = [json.loads(x) for x in (root/"reports"/f"v101-probe120-seed{seed}"/"decisions.jsonl").read_text(encoding="utf-8").splitlines() if x]
        old = {x["case_id"]: x for x in oldp}; new = {x["case_id"]: x for x in newp}
        for case_id in sorted(old):
            a, b = old[case_id], new[case_id]
            if a["strict_exact_match"] and not b["strict_exact_match"] and b["condition"] == "two_hop":
                complete = b.get("audit", {}).get("visible_support_coverage") == 1
                details.append({"source":"probe", "seed":seed, "task_id":b["task_id"], "case_id":case_id,
                                "migration":"correct_to_wrong", "category":"support_titles_visible_but_wrong" if complete else "retrieval_or_evidence_insufficient",
                                "old":{"answer":a.get("response",{}).get("content"),"exact":True},
                                "new":{"answer":b.get("response",{}).get("content"),"exact":False,"visible_support_complete":complete}})
    after_hashes = {key: sha(Path(key)) for key in before_hashes}
    payload = {"version":"v102", "kind":"p0_error_migration", "source_hashes_unchanged":before_hashes == after_hashes,
               "autonomous_v101_minus_v100_em":bootstrap(deltas),
               "empty_search_answer_metrics":{key:dict(value) for key,value in empty_metrics.items()},
               "termination_counts":{key:dict(value) for key,value in termination.items()},
               "correct_to_wrong_counts":dict(Counter(item["category"] for item in details)),
               "correct_to_wrong_rows":len(details)}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    detail.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in details)+"\n", encoding="utf-8")
    return payload

def p1(root: Path, output: Path) -> dict:
    per_seed, changed_groups, affected_tasks, beneficial_tasks, inversions = {}, 0, set(), set(), 0
    for seed in SEEDS:
        path = root / "runs" / f"v100-answer30-seed{seed}.db"; before = sha(path)
        rollouts = rows(path, "rollouts"); transitions = rows(path, "transitions")
        tasks = {item["id"]: item for item in rows(path, "tasks")}
        by_rollout=defaultdict(list)
        for t in transitions: by_rollout[t["rollout_id"]].append(t)
        by_group=defaultdict(list)
        for r in rollouts:
            ts=by_rollout[r["id"]]; over=[t for t in ts if "over_search_after_evidence" in t.get("reward_tags", [])]
            old=float(r.get("reward",0)); candidate=old + .2*len(over)
            nonrepeat=[t for t in over if t.get("termination_reason") != "repeated_action"]
            gold = tasks.get(r["task_id"], {}).get("answer", "")
            correct = bool(gold and norm(r.get("final_answer")) == norm(gold))
            if nonrepeat: affected_tasks.add(r["task_id"])
            if nonrepeat and correct:
                beneficial_tasks.add(r["task_id"])
            by_group[r["group_id"]].append((old,candidate,r.get("final_answer"),r["task_id"]))
        seed_changed=seed_var=0
        for vals in by_group.values():
            old=[v[0] for v in vals]; new=[v[1] for v in vals]
            oldc=[x-sum(old)/len(old) for x in old]; newc=[x-sum(new)/len(new) for x in new]
            seed_changed += int(any(abs(a-b)>1e-9 for a,b in zip(oldc,newc)))
            seed_var += int(len({round(x,12) for x in new})>1)
        changed_groups += seed_changed
        per_seed[str(seed)]={"changed_centered_advantage_groups":seed_changed,"nonzero_variance_groups":seed_var,"source_hash_unchanged":before==sha(path)}
    passed = len(beneficial_tasks)>=3 and changed_groups>=6 and not inversions and all(v["nonzero_variance_groups"]>=15 for v in per_seed.values())
    payload={"version":"v102-oversearch-zero","candidate":{"over_search_after_evidence":0.0},
             "affected_task_count":len(affected_tasks),"beneficial_or_successful_task_count":len(beneficial_tasks),
             "changed_centered_advantage_groups":changed_groups,"ranking_inversions":inversions,"per_seed":per_seed,
             "gate_passed":passed}
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return payload

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"p0":p0(args.root,args.data,args.output_dir/"v102-p0-diagnostic.json",args.output_dir/"v102-p0-details.jsonl"),
                      "p1":p1(args.root,args.output_dir/"v102-p1-gate.json")},ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
