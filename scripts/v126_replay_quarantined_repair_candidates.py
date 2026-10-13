"""Replay two v117 quarantine repair candidates without creating labels."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from transformers import AutoProcessor

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.training.sft import encode_assistant_only
from scripts.prepare_v116_review import PROCESSOR, render
from scripts.v123_cpu_core import ROOT, digest, readl, write, writel
from scripts.v124_forensic import OUT


TARGETS = {
    ("v117_train:0001", "v117_train:0167"): "According to the cited First Lady of Ghana source, who was identified as the current holder of the title previously held by Nana Konadu Agyeman-Rawlings?",
    ("v117_train:0050", "v117_train:0161"): "What population figure does the Rimini article report for the city near Alberta Ferretti's studio?",
}


def main() -> None:
    out_path = OUT / "v126-quarantined-repair-replay-candidates-private.jsonl"
    report_path = OUT / "v126-quarantined-repair-replay-report-private.json"
    manifest_path = OUT / "v126-quarantined-repair-replay-manifest-private.json"
    if any(p.exists() for p in (out_path, report_path, manifest_path)):
        raise FileExistsError("refuse_to_overwrite_v126_quarantine_replay")

    source_path = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
    trajectory_path = OUT / "repair-draft-trajectories-cpu-private.jsonl"
    inventory_path = OUT / "inventory-private.jsonl"
    link_path = OUT / "v125-v117-repair-trajectory-link-audit-private.json"
    originals = readl(source_path)
    by_id = {f"v117_train:{index:04d}": row for index, row in enumerate(originals, 1)}
    inventory = {r["record_id"]: r for r in readl(inventory_path)}
    groups = readl(trajectory_path)
    group_by_pair = {tuple(g["record_ids"]): g for g in groups}
    if not set(TARGETS) <= set(group_by_pair):
        raise ValueError("expected_quarantine_repair_groups_missing")

    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    output, summary = [], []
    for ids, question in TARGETS.items():
        group = group_by_pair[ids]
        states = []
        for record_id in ids:
            if record_id not in by_id or record_id not in inventory:
                raise ValueError("source_record_missing")
            original = by_id[record_id]
            if digest(original) != inventory[record_id]["source_row_sha256"]:
                raise ValueError("original_source_hash_mismatch")
            draft_state = next(s for s in group["states"] if s["record_id"] == record_id)
            row = json.loads(json.dumps(original, ensure_ascii=False))
            row["messages"][1]["content"] = question
            target_index = row["target_assistant_index"]
            action = draft_state["proposed_action"]
            row["messages"][target_index]["content"] = json.dumps(
                action, ensure_ascii=False, separators=(",", ":"))
            prompt = row["messages"][:target_index]
            prompt_ids = render(processor, [SimpleNamespace(**m) for m in prompt])
            target = {"role": "assistant", "content": row["messages"][target_index]["content"]}
            encoded = encode_assistant_only(processor, prompt + [target], 2176,
                                            len(prompt), False)
            target_tokens = sum(token != -100 for token in encoded["labels"])
            within = len(prompt_ids) <= 2048 and target_tokens <= 128 and len(encoded["input_ids"]) <= 2176
            states.append({
                "record_id_private": record_id,
                "original_row_sha256": digest(original),
                "revised_row_sha256": digest(row),
                "question": question,
                "visible_messages": prompt,
                "before_action_evidence": visible_documents(prompt),
                "proposed_action": action,
                "retrieved_after_search": next(
                    (s["retrieved_after_search"] for s in group["states"]
                     if s["record_id"] == record_id), None),
                "prompt_tokens": len(prompt_ids), "target_tokens": target_tokens,
                "total_tokens": len(encoded["input_ids"]), "within_token_limits": within,
                "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                "target_mask_sha256": hashlib.sha256(json.dumps(encoded["labels"]).encode()).hexdigest(),
                "training_allowed": False,
            })
        action_set = {s["proposed_action"]["action"] for s in states}
        pair = {"question": question, "record_ids_private": list(ids), "states": states,
                "both_search_and_answer": action_set == {"search", "answer"},
                "token_limits_passed": all(s["within_token_limits"] for s in states),
                "status": "rewrite_candidate_requires_fresh_independent_double_review",
                "training_allowed": False}
        output.append(pair)
        summary.append({"record_ids_private": list(ids),
            "both_search_and_answer": pair["both_search_and_answer"],
            "token_limits_passed": pair["token_limits_passed"],
            "search_query_changed": False,
            "status": pair["status"], "training_allowed": False})

    link_report = json.loads(link_path.read_text(encoding="utf-8"))
    manifest = {
        "version": "v126-quarantined-repair-replay-v1",
        "inputs": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (source_path, trajectory_path, inventory_path, link_path)},
        "processor_path": str(PROCESSOR), "local_files_only": True,
        "states": sum(len(g["states"]) for g in output), "questions": len(output),
        "search_queries_authored_or_changed": False,
        "replay_scope": "rewrite and tokenize replay of only the two already strict-linked quarantined v117 groups",
        "source_link_report_sha256": link_report["inputs"][str(trajectory_path.relative_to(ROOT)).replace("\\", "/")],
        "training_allowed": False,
    }
    writel(out_path, output)
    write(manifest_path, manifest)
    report = {"version": "v126-quarantined-repair-replay-report-v1",
        "groups": len(output), "states": sum(len(g["states"]) for g in output),
        "summary": summary, "all_original_rows_unchanged": True,
        "independent_semantic_reviews": 0, "training_allowed": False}
    write(report_path, report)
    print(json.dumps({"groups": report["groups"], "states": report["states"],
        "all_token_limits_pass": all(g["token_limits_passed"] for g in output),
        "independent_semantic_reviews": 0, "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
