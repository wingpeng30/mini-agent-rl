"""Rebuild v115 blind packets with candidate- and step-bound evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docs(messages):
    """Only text actually present in tool messages; compact ref stubs add no text."""
    found = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            observation = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            continue
        for doc in observation.get("result", []) if isinstance(observation, dict) else []:
            if isinstance(doc, dict) and doc.get("id") and isinstance(doc.get("text"), str) and doc["text"]:
                found[doc["id"]] = {"id": doc["id"], "title": doc.get("title", ""), "text": doc["text"]}
    return list(found.values())


def result_docs(observation):
    if not isinstance(observation, dict):
        return None
    result = observation.get("result")
    if not isinstance(result, list):
        return None
    return [{"id": d["id"], "title": d.get("title", ""), "text": d["text"]}
            for d in result if isinstance(d, dict) and d.get("id") and isinstance(d.get("text"), str) and d["text"]]


def fixed_packets(packet_dir: Path, report_dir: Path, out: Path):
    old_manifest = read(packet_dir / "manifest.json")
    if sha(packet_dir / "states-blind.jsonl") != old_manifest["state_packet_sha256"]:
        raise ValueError("source_state_packet_changed")
    if sha(packet_dir / "changed-tasks-blind.jsonl") != old_manifest["changed_task_packet_sha256"]:
        raise ValueError("source_task_packet_changed")
    mapping = read(packet_dir / "private-mapping.json")
    if sha(packet_dir / "private-mapping.json") != old_manifest["private_mapping_sha256"]:
        raise ValueError("private_mapping_changed")
    report_paths = {model: report_dir / f"v115-det-{model}-states.json" for model in ("v101", "v112", "v115")}
    reports = {model: {r["state_id"]: r for r in read(path)["rows"]} for model, path in report_paths.items()}
    state_packets = []
    for packet in rows(packet_dir / "states-blind.jsonl"):
        aid = packet["audit_id"]
        source = mapping["states"][aid]
        visible = docs(packet["visible_history_and_evidence"])
        candidates = {}
        for alias, output in packet["anonymous_outputs"].items():
            report = reports[source["anonymous_to_model"][alias]][source["state_id"]]
            action = report.get("actual_action")
            if output.get("action") != action:
                raise ValueError("anonymous_action_mismatch:" + aid)
            retrieved = result_docs({"result": report.get("retrieval_results")}) if action and action.get("action") == "search" else None
            candidates[alias] = {"action": action, "raw_output": output.get("raw_output", ""),
                                 "before_action": visible, "retrieved_after_search": retrieved}
        state_packets.append({"audit_id": aid, "question": packet["question"], "kind": "state",
                              "candidates": candidates})
    task_packets = []
    for packet in rows(packet_dir / "changed-tasks-blind.jsonl"):
        candidates = {}
        for alias, output in packet["anonymous_outputs"].items():
            steps = []
            for transition in output.get("trajectory", []):
                response = transition.get("output") or {}
                observation = transition.get("observation") or {}
                steps.append({"step": transition["step"], "before_action": docs(transition["visible_messages"]),
                              "output": response, "retrieved_after_search": result_docs(observation)
                              if response.get("tool_calls") else None})
            candidates[alias] = {"final_answer": output.get("final_answer"), "steps": steps}
        task_packets.append({"audit_id": packet["audit_id"], "question": packet["question"],
                             "kind": "changed_task", "candidates": candidates})
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("states", state_packets), ("changed-tasks", task_packets)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                                                   for r in data), encoding="utf-8")
    manifest = {"version": "v116-blind-diagnosis-v1", "states": len(state_packets),
                "changed_tasks": len(task_packets), "aliases_per_case": 3,
                "source_manifest_sha256": sha(packet_dir / "manifest.json"),
                "state_reports_sha256": {m: sha(p) for m, p in report_paths.items()},
                "states_sha256": sha(out / "states.jsonl"),
                "changed_tasks_sha256": sha(out / "changed-tasks.jsonl"),
                "blinded": True, "gold_in_packets": False, "model_identity_in_packets": False}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--packets", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(fixed_packets(a.packets, a.reports, a.out), ensure_ascii=False))
