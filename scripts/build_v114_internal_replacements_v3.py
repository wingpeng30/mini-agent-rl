"""Freeze three independently approved internal question replacements and their review packets."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoProcessor

from build_v114_sft_candidates import OUT as SOURCE, ROOT, build_examples, build_index, load_full, norm, public_example
from prepare_v114_reviews import MODEL, canonical, sha
from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import budget_messages
from mini_agent_rl.training.sft import encode_assistant_only

WORK = ROOT / "data/evidence-sft-v114-review"
REMOVED_IDS = {
    "5ab33ee955429969a97a810d",  # Kokomo / WWI source evidence defect
    "5ac3d8bd554299204fd21e9e",  # Cowboy Junkies / comparative ambiguity
    "5ae61243554299546bf83063",  # Mac and Me / identity and attribution defect
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def freeze(path: Path, data: bytes):
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f"frozen_artifact_changed:{path.name}")
    if not path.exists():
        path.write_bytes(data)


def main():
    old = read_json(WORK / "internal-freeze-v2.json")
    a = read_jsonl(WORK / "internal-replacement-proposals-a-decisions.jsonl")
    b = read_jsonl(WORK / "internal-replacement-proposals-b-decisions.jsonl")
    qa = {row["audit_id"]: row for row in a}
    qb = {row["audit_id"]: row for row in b}
    private = read_jsonl(WORK / "internal-replacement-proposals-private.jsonl")
    if len(a) != 12 or len(b) != 12 or len(qa) != 12 or len(qb) != 12:
        raise RuntimeError("replacement_question_audit_incomplete")
    if [r["audit_id"] for r in a] != [r["audit_id"] for r in b]:
        raise RuntimeError("replacement_question_audit_order_mismatch")
    if any(qa[k]["packet_sha256"] != qb[k]["packet_sha256"] for k in qa):
        raise RuntimeError("replacement_question_audit_packet_mismatch")
    source_order = {row["audit_id"]: row for row in private}
    proposal_packets_a = {r["audit_id"]: r for r in read_jsonl(WORK / "internal-replacement-proposals-a-packets.jsonl")}
    proposal_packets_b = {r["audit_id"]: r for r in read_jsonl(WORK / "internal-replacement-proposals-b-packets.jsonl")}
    selected = []
    for row in a:
        aid = row["audit_id"]
        pa, pb = proposal_packets_a.get(aid), proposal_packets_b.get(aid)
        if not pa or pa != pb or qa[aid]["packet_sha256"] != pa.get("packet_sha256"):
            raise RuntimeError("replacement_question_packet_binding_invalid")
        for decision in (qa[aid], qb[aid]):
            context = {item["title"]: item["text"] for item in pa["context"]}
            citations = decision.get("citations")
            if not isinstance(citations, list) or not citations:
                raise RuntimeError("replacement_question_audit_missing_citations")
            if any(item.get("title") not in context or not item.get("quote")
                   or item["quote"] not in context[item["title"]] for item in citations):
                raise RuntimeError("replacement_question_audit_illegal_citation")
            reason = decision.get("reason", "")
            # Any quoted phrase in the rationale must be traceable to this packet.
            for match in re.finditer(r'“([^”]+)”|"([^"]+)"', reason):
                quote = match.group(1) or match.group(2)
                if quote and not any(quote in text for text in context.values()):
                    raise RuntimeError("replacement_question_audit_reason_quote_unbound")
        if qa[aid]["verdict"] == "usable" and qb[aid]["verdict"] == "usable":
            selected.append(source_order[aid])
            if len(selected) == 3:
                break
    if len(selected) != 3:
        raise RuntimeError("fewer_than_three_dual_usable_replacement_questions")
    replace_old = {"5ab33ee955429969a97a810d", "5ac3d8bd554299204fd21e9e", "5ae61243554299546bf83063"}
    old_ids = set(old["ids"])
    if not replace_old <= old_ids or len({r["source_id"] for r in selected}) != 3:
        raise RuntimeError("replacement_source_selection_invalid")
    new_source_ids = {r["source_id"] for r in selected}
    new_ids = (old_ids - replace_old) | new_source_ids
    if len(new_ids) != 50 or new_ids & set(read_json(SOURCE / "training-question-freeze.json")["ids"]):
        raise RuntimeError("replacement_freeze_size_or_isolation_invalid")

    # Verify selected question/source hashes are bound to the double-reviewed question packets.
    for item in selected:
        aid = item["audit_id"]
        if proposal_packets_a[aid] != proposal_packets_b[aid] or qa[aid]["packet_sha256"] != proposal_packets_a[aid]["packet_sha256"]:
            raise RuntimeError("replacement_question_packet_binding_invalid")
        if hashlib.sha256(norm(proposal_packets_a[aid]["question"]).encode()).hexdigest() != item["normalized_question_sha256"]:
            raise RuntimeError("replacement_question_hash_mismatch")

    all_source_rows = load_full(new_ids)
    train_freeze = read_json(WORK / "training-question-freeze-v2.json")
    train_hashes = set(train_freeze["question_hashes"])
    question_hashes = {hashlib.sha256(norm(row["question"]).encode()).hexdigest() for row in all_source_rows.values()}
    if question_hashes & train_hashes:
        raise RuntimeError("replacement_freeze_normalized_question_overlap")
    risk_counts = {}
    from build_v114_sft_candidates import risk_tags
    for row in all_source_rows.values():
        for tag in risk_tags(row["question"]):
            risk_counts[tag] = risk_counts.get(tag, 0) + 1
    freeze_obj = {
        "version": "v114-internal-freeze-v3",
        "parent_freeze_sha256": sha((WORK / "internal-freeze-v2.json").read_bytes()),
        "question_count": 50,
        "ids": sorted(new_ids),
        "question_hashes": sorted(question_hashes),
        "risk_question_counts": risk_counts,
        "replaced_source_ids": sorted(replace_old),
        "replacement_source_ids": [r["source_id"] for r in selected],
        "replacement_audit_ids": [r["audit_id"] for r in selected],
        "selection_rule": "remove only the three dual-confirmed ambiguous/source-defective questions; add the first three unique proposals in fixed packet order that both blinded auditors rated usable; no outcomes or labels used",
        "independent100_content_read": False,
        "final_test_accessed": False,
    }
    freeze(WORK / "internal-freeze-v3.json", json.dumps(freeze_obj, ensure_ascii=False, indent=2).encode() + b"\n")

    source_rows = {key: all_source_rows[key] for key in new_source_ids}
    _, index = build_index(list(all_source_rows.values()))
    examples = build_examples(list(source_rows.values()), index)
    if not examples:
        raise RuntimeError("replacement_questions_produced_no_states")
    # Keep every distinct state; no outcome-driven selection is applied.
    examples.sort(key=lambda e: (e["qid"], e["state_kind"], hashlib.sha256(
        json.dumps(e["action"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()))
    public = [public_example(example, i) for i, example in enumerate(examples)]
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    packets, keys = [], []
    for row, example in zip(public, examples):
        messages = [Message(**m) for m in row["visible_messages"]]
        def render(ms):
            result = processor.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in ms],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if isinstance(result, dict):
                result = result["input_ids"]
            return result[0] if len(result) == 1 and isinstance(result[0], list) else result
        budgeted = budget_messages(messages, 2048, lambda ms: len(render(ms)),
                                   lambda s: len(processor.tokenizer.encode(s, add_special_tokens=False)),
                                   observation_encoding="compact-v3")
        actual = [{"role": m.role, "content": m.content} for m in budgeted.messages]
        token_ids = render(budgeted.messages)
        target = {"role": "assistant", "content": json.dumps(row["proposed_action"], ensure_ascii=False, separators=(",", ":"))}
        encoded = encode_assistant_only(processor, actual + [target], 2176, len(actual), False)
        target_tokens = sum(value != -100 for value in encoded["labels"])
        if len(token_ids) > 2048 or target_tokens > 128:
            raise RuntimeError("replacement_candidate_exceeds_frozen_token_budget")
        docs = visible_documents(actual)
        packet = {"case_id": row["case_id"], "question": row["question"], "visible_messages": actual,
                  "proposed_action": row["proposed_action"], "visible_evidence": docs,
                  "retrieval_top3": row["retrieval_top3"], "prompt_sha256": sha(bytes(str(token_ids), "utf-8")),
                  "prompt_tokens": len(token_ids), "target_tokens": target_tokens, "preflight_error": None}
        packet["packet_sha256"] = sha(canonical(packet))
        packets.append(packet)
        keys.append({"case_id": row["case_id"], "source_id": example["qid"], "split": "internal",
                     "state_kind": example["state_kind"], "risk_tags": example["risk_tags"]})
    if len({r["case_id"] for r in packets}) != len(packets):
        raise RuntimeError("replacement_state_duplicate")
    payload = b"".join(canonical(row) + b"\n" for row in packets)
    for reviewer in ("a", "b"):
        freeze(WORK / f"internal-replacements-v3-{reviewer}-packets.jsonl", payload)
    freeze(WORK / "internal-replacements-v3-private.jsonl", b"".join(canonical(row) + b"\n" for row in keys))
    manifest = {"version": "v114-internal-replacements-v3", "state_count": len(packets),
                "source_ids": [r["source_id"] for r in selected], "audit_ids": [r["audit_id"] for r in selected],
                "packet_sha256": sha(payload), "freeze_sha256": sha((WORK / "internal-freeze-v3.json").read_bytes()),
                "private_mapping_sha256": sha((WORK / "internal-replacements-v3-private.jsonl").read_bytes()),
                "independent100_content_read": False, "final_test_accessed": False}
    freeze(WORK / "internal-replacements-v3-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"replacements": [r["source_id"] for r in selected], "states": len(packets),
                      "state_kinds": {kind: sum(r["state_kind"] == kind for r in keys) for kind in
                                      ("supported_answer", "nonempty_evidence_search", "empty_evidence_search")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
