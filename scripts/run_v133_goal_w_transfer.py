"""Goal W orchestration and data-quota stop, using frozen source artifacts.

The run must stop before GPU training if the one-time candidate pool cannot
meet its four-category evidence-pair quota. `summarize` is CPU-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-w-transfer"
DATA = OUT / "data"
STATE = OUT / "state.json"
CONFIG = OUT / "config.json"
AUDIT = OUT / "affiliation-question-audit.jsonl"

# Deliberately generous upper bound. These questions might ask about a group
# shared by two subjects; they still need evidence and negative-pair review.
# All other affiliation-tagged training questions clearly ask for another
# attribute (person, date, age, occupation, etc.).
POSSIBLE_AFFILIATION = {
    "5a835f305542992ef85e2298",  # country of a shared party
    "5ab2d8d55542991669774108",  # court hearing two cases
    "5ab3f12f55429976abd1bcfe",  # same musical group
    "5adcdde65542992c1e3a243d",  # awarding group for two devices
    "5addedf15542997dc79070bc",  # person shared by two organizations
    "5ac2c449554299218029db32",  # wrestling organization
    "5a791cbe55429907847277be",  # shared sports organization
    "5ac402535542997ea680c9b1",  # shared band member
    "5abb026d5542996cc5e49f4b",  # shared political party
    "5ab92fbe554299131ca422ab",  # shared band member
    "5ab2a6655542992953946796",  # shared band
    "5a754a775542993748c89818",  # member of two groups
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def baseline_counts():
    data = rows(OUT / "w0-old-baseline.jsonl")
    if len(data) != 32 or len({v["case_id"] for v in data}) != 32:
        raise RuntimeError("W0 baseline incomplete")
    return {"total": sum(v["em"] for v in data),
            "regression": sum(v["em"] for v in data if v["split"] == "goal_G_regression"),
            "invalid": sum(v["reason"] != "answer" for v in data)}


def verify_data_manifest():
    manifest = read(DATA / "manifest.json")
    if manifest["status"] != "bounded_unreviewed_candidate_pools_prepared":
        raise RuntimeError("Candidate pool was not frozen")
    if manifest["historical_exclusions"]["reconstructed_goal_v"]["task_id_sha256"] != \
            "c8f575d8d55cb32937d1b649213f81c20dbaaa007e87d291ef8c9d8c11e89432":
        raise RuntimeError("Sealed exclusion reconstruction changed")
    for name, item in manifest["outputs"].items():
        path = DATA / name
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen data changed: {name}")
    for split, size in (("train", 512), ("dev", 128), ("confirm", 128)):
        if manifest["candidate_prefilter"]["selected_counts"][split] != size:
            raise RuntimeError(f"Unfilled candidate pool: {split}")
        if manifest["corpora"]["details"]["documents"][split]["documents"] != 2048:
            raise RuntimeError(f"Unfilled corpus: {split}")
    return manifest


def prepare():
    if CONFIG.exists() or STATE.exists():
        raise FileExistsError("Goal W already prepared")
    manifest = verify_data_manifest()
    baseline = baseline_counts()
    if baseline != {"total": 20, "regression": 7, "invalid": 0}:
        raise RuntimeError(f"Goal T old baseline changed: {baseline}")
    old = read(ROOT / "runs/v133-goal-u-v-evidence-preference/config.json")
    write(CONFIG, {
        "version": "goal-w-transfer-v1", "protocol_status": "source_and_baseline_frozen",
        "data_manifest_sha256": sha(DATA / "manifest.json"),
        "attempt01_manifest_sha256": sha(OUT / "data-attempt01/manifest.json"),
        "pre_model_correction_sha256": sha(OUT / "pre-model-correction.json"),
        "baseline_sha256": sha(OUT / "w0-old-baseline.jsonl"),
        "baseline": baseline,
        "source_adapter_weight_sha256": old["files"]["adapter"],
        "base_weight_sha256": old["files"]["base_0"],
        "tokenizer_sha256": old["files"]["tokenizer"],
        "candidate_counts": manifest["candidate_prefilter"]["selected_counts"],
        "category_quota": {"train": 16, "dev": 8, "confirm": 8},
        "planned_seeds": [42, 43], "planned_updates_per_arm": 128,
        "planned_arms": ["chosen_sft", "dpo"],
        "gpu_time_limit": None,
    })
    write(STATE, {"status": "prepared", "gpu_training_started": False,
                  "stages": [{"name": "W0", "status": "passed"}], "decision": None})
    return {"baseline": baseline, "candidate_counts": manifest["candidate_prefilter"]["selected_counts"]}


def run():
    config = read(CONFIG)
    state = read(STATE)
    if state["status"] != "prepared":
        raise RuntimeError("Goal W run is not in prepared state")
    if sha(DATA / "manifest.json") != config["data_manifest_sha256"]:
        raise RuntimeError("Frozen candidate manifest changed")
    data = rows(DATA / "candidate-train.jsonl")
    affiliation = [x for x in data if x["attribute_category"] == "affiliation"]
    ids = {x["source_id"] for x in affiliation}
    if len(data) != 512 or len(affiliation) != 28 or not POSSIBLE_AFFILIATION.issubset(ids):
        raise RuntimeError("Question audit does not match frozen candidates")
    audit = []
    for item in affiliation:
        plausible = item["source_id"] in POSSIBLE_AFFILIATION
        audit.append({"case_id": item["case_id"], "question": item["question"],
                      "question_sha256": hashlib.sha256(item["question"].encode()).hexdigest(),
                      "classification": "possibly_shared_group_relation" if plausible else "different_asked_attribute",
                      "reason": ("Generous upper-bound inclusion; still needs two-sided evidence and a same-attribute negative."
                                 if plausible else
                                 "The original question asks for a person, date, occupation, age, work, or another relation rather than a common affiliation attribute of the two named subjects.")})
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    AUDIT.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in audit), encoding="utf-8")
    possible = sum(x["classification"] == "possibly_shared_group_relation" for x in audit)
    if possible >= config["category_quota"]["train"]:
        raise RuntimeError("Data gate now appears feasible; implement the remaining scheduled stages before GPU work")
    decision = {"selected": "goal_t_seed42_update32", "reason": "W1_train_affiliation_quota_shortfall",
                "status": "stopped_before_new_model_work", "minimum_required": 16,
                "generous_upper_bound": possible, "source_candidates": len(affiliation),
                "audit_sha256": sha(AUDIT), "baseline": config["baseline"]}
    write(OUT / "decision.json", decision)
    state.update(status="complete_data_shortfall", decision=decision)
    state["stages"].append({"name": "W1", "status": "failed_quota", "eligible_upper_bound": possible})
    STATE.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return decision


def summarize():
    cfg, state = read(CONFIG), read(STATE)
    if sha(DATA / "manifest.json") != cfg["data_manifest_sha256"]:
        raise RuntimeError("Frozen candidate manifest changed")
    if baseline_counts() != cfg["baseline"]:
        raise RuntimeError("W0 baseline changed")
    result = {"status": state["status"], "baseline": cfg["baseline"],
              "decision": state["decision"], "candidate_counts": cfg["candidate_counts"],
              "training_started": state["gpu_training_started"]}
    if state["status"] == "complete_data_shortfall":
        votes = rows(AUDIT)
        if len(votes) != 28 or sha(AUDIT) != state["decision"]["audit_sha256"]:
            raise RuntimeError("Question audit changed")
        result["question_audit"] = dict(Counter(x["classification"] for x in votes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    command = parser.parse_args().command
    print(json.dumps({"prepare": prepare, "run": run, "summarize": summarize}[command](), ensure_ascii=False))


if __name__ == "__main__":
    main()
