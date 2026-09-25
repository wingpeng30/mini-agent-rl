"""Select all v115 cases affected by candidate evidence errors or unresolved votes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "reports/v115-private-run"
FIXED = ROOT / "data/evidence-sft-v116/diagnosis-blind"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def evidence(case, alias):
    output = case["candidates"][alias]
    if case["kind"] == "state":
        before = output["before_action"]
        after = output["retrieved_after_search"] or []
    else:
        before = [d for step in output["steps"] for d in step["before_action"]]
        after = [d for step in output["steps"] for d in (step["retrieved_after_search"] or [])]
    return before, after


def main():
    reviewers = [load(OLD / f"reviewer-{name}.json") for name in ("a", "b")]
    decisions = [{r["audit_id"]: r for section in ("states", "changed_tasks") for r in reviewer[section]}
                 for reviewer in reviewers]
    unresolved = {r["audit_id"] for r in load(OLD / "semantic-merge.json")["unresolved_cases"]}
    source = rows(FIXED / "states.jsonl") + rows(FIXED / "changed-tasks.jsonl")
    selected, reasons = [], {}
    for case in source:
        aid = case["audit_id"]
        causes = ["old_unresolved"] if aid in unresolved else []
        for alias in "ABC":
            before, after = evidence(case, alias)
            before_text = "\n".join(d["text"] for d in before)
            after_text = "\n".join(d["text"] for d in after)
            for reviewer, by_case in zip(("a", "b"), decisions):
                for quote in by_case[aid]["candidates"][alias].get("citations", []):
                    if quote not in before_text:
                        causes.append("old_quote_future_only" if quote in after_text else "old_quote_not_candidate_visible")
        if causes:
            selected.append(case)
            reasons[aid] = sorted(set(causes))
    if len({r["audit_id"] for r in selected}) != len(selected):
        raise ValueError("duplicate_rereview_id")
    out = FIXED / "rereview";out.mkdir(parents=True, exist_ok=True)
    path = out / "selected.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                            for r in selected), encoding="utf-8")
    for start in range(0, len(selected), 10):
        (out / f"batch-{start // 10 + 1:02}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for r in selected[start:start + 10]), encoding="utf-8")
    manifest = {"version": "v116-diagnosis-rereview-v1", "count": len(selected),
                "states": sum(r["kind"] == "state" for r in selected),
                "changed_tasks": sum(r["kind"] == "changed_task" for r in selected),
                "source_manifest_sha256": hashlib.sha256((FIXED / "manifest.json").read_bytes()).hexdigest(),
                "selected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reasons": reasons}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "reasons"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
