"""Score the corrected AP v2 one-pass adjudication revision without replacing the first gate record."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/v133-goal-ap-evidence-presentation-v2"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_once(path, obj):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def main():
    rows = read_jsonl(RUN / "probe-adjudications-revision.jsonl")
    by = {(x["model"], x["arm"], x["probe_id"]): x for x in rows}

    def metric(model, arm, category):
        selected = [x for x in rows if (x["model"], x["arm"], x["category"]) == (model, arm, category)]
        return {"correct": sum(x.get("supported_correct") is True for x in selected),
                "binding_repairs": sum(x.get("binding_repair") is True for x in selected),
                "legal_search": sum(x.get("legal_bridge_search") is True for x in selected),
                "premature": sum(x.get("premature_answer") is True for x in selected),
                "invalid": sum(x.get("invalid_action") is True for x in selected),
                "repeated": sum(x.get("repeated_query") is True for x in selected),
                "count": sum(x.get("supported_correct") is not None for x in selected)}

    a, b = metric("AM", "A", "complete"), metric("AM", "B", "complete")
    pa, pb = metric("AM", "A", "partial"), metric("AM", "B", "partial")
    losses = sum(by[("AM", "A", pid)]["supported_correct"] is True and
                 by[("AM", "B", pid)]["supported_correct"] is not True
                 for pid in {x["probe_id"] for x in rows if x["model"] == "AM" and x["category"] == "complete"})
    rules = read_json(RUN / "manifest.json")["gate"]
    passed = (b["correct"] >= rules["complete_B_at_least"] and
              b["correct"] - a["correct"] >= rules["complete_gain_vs_A"] and
              b["binding_repairs"] >= rules["binding_repair_min"] and
              losses <= rules["complete_losses_max"] and
              pb["legal_search"] >= rules["partial_legal_search_min"] and
              pb["legal_search"] >= pa["legal_search"] and
              pb["premature"] <= pa["premature"] and pb["invalid"] <= pa["invalid"] and
              pb["repeated"] <= pa["repeated"])
    revision = {"passed": passed, "am": {"A_complete": a, "B_complete": b,
                                            "A_partial": pa, "B_partial": pb,
                                            "A_correct_states_lost": losses},
                "z": "descriptive only; not counted in continuation gate",
                "gate_rules": rules,
                "adjudications_sha256": hashlib.sha256((RUN / "probe-adjudications-revision.jsonl").read_bytes()).hexdigest(),
                "parent_gate_sha256": hashlib.sha256((RUN / "probe-gate.json").read_bytes()).hexdigest(),
                "reason_for_revision": "The first partial-query adjudication treated a year appearing in an unrelated UEFA Europa League passage as relevant to the requested FA Cup win relation. The corrected ruling rejects that query's unsupported year narrowing. Both gates fail the complete-evidence threshold; this correction does not alter the stop decision."}
    write_once(RUN / "probe-gate-revision.json", revision)
    amendment = {"status": "scoring_revision_only", "parent_adjudication_sha256": hashlib.sha256(
                    (RUN / "probe-adjudications-initial.jsonl").read_bytes()).hexdigest(),
                 "revision_adjudication_sha256": revision["adjudications_sha256"],
                 "parent_gate_sha256": revision["parent_gate_sha256"],
                 "revision_gate_sha256": hashlib.sha256((RUN / "probe-gate-revision.json").read_bytes()).hexdigest(),
                 "change": "Crossley partial B query remains ineligible because 2015–16 appears only in an unrelated UEFA Europa League rule and does not support the missing FA Cup win-date relation.",
                 "decision_changed": False}
    write_once(RUN / "probe-adjudication-amendment.json", amendment)
    print(json.dumps({"passed": passed, "A_complete": a["correct"], "B_complete": b["correct"],
                      "A_to_B_gain": b["correct"] - a["correct"], "A_correct_states_lost": losses,
                      "A_partial_legal": pa["legal_search"], "B_partial_legal": pb["legal_search"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
