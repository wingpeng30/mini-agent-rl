"""Freeze Goal G's separately reviewed, project-isolated CPU evaluation subset."""
from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path

from scripts.run_v133_goal_g_cpu import OUT, ROOT, EXPOSURE_FILES, make_case, readl, sha, write, writel
from scripts.v130_review_protocol import valid_vote

EXPECTED_PACKET_SHA = "c5c44da23a275e76089f67d93b39e25e0e1e9a706628a317007f501aa7e4da26"
BATCHES = ("g-00-04", "g-05-09", "g-10-14", "g-15-18")


def two_document_citations(packet, vote, luna=False):
    docs = {d["id"]: d["text"] for d in packet["before_action"]}
    seen = set()
    for citation in vote.get("citations", []):
        if not luna and (citation.get("stage"), citation.get("step")) != ("before_action", None):
            raise ValueError("deepseek_citation_outside_before_action")
        document_id, quote = citation.get("document_id"), citation.get("quote")
        if document_id not in docs or not isinstance(quote, str) or not quote.strip() or quote not in docs[document_id]:
            raise ValueError("citation_not_exactly_visible")
        seen.add(document_id)
    if vote["approved"] and seen != set(docs):
        raise ValueError("approved_vote_not_supported_by_both_documents")


def main():
    if (OUT / "evaluation-frozen.jsonl").exists():
        raise FileExistsError("evaluation_already_frozen")
    if sha(OUT / "review-packets.jsonl") != EXPECTED_PACKET_SHA:
        raise ValueError("review_packet_hash_mismatch")
    packets = list(readl(OUT / "review-packets.jsonl"))
    candidates = list(readl(OUT / "review-candidates.jsonl"))
    if len(packets) != len(candidates) != 19:
        raise ValueError("packet_candidate_count_mismatch")
    packet_by_id = {p["case_id"]: p for p in packets}
    if set(packet_by_id) != {f"g{i:03d}" for i in range(19)}:
        raise ValueError("case_id_binding_mismatch")
    cpu = {x["source_id"]: x for x in readl(OUT / "cpu-preflight-v2.jsonl")}
    ancestry = json.loads((OUT / "ancestry-audit.json").read_text(encoding="utf-8"))
    if len(ancestry["rows"]) != 19:
        raise ValueError("ancestor_audit_count_mismatch")
    luna_list = json.loads((OUT / "luna-votes-technical-correction.json").read_text(encoding="utf-8"))
    luna = {x["case_id"]: x for x in luna_list}
    if len(luna) != 19 or set(luna) != set(packet_by_id):
        raise ValueError("luna_vote_binding_mismatch")
    deepseek = {}
    batch_hashes = {}
    for name in BATCHES:
        directory = OUT / "deepseek" / name
        manifest = json.loads((directory / "manifest-private.json").read_text(encoding="utf-8"))
        identity = json.loads((directory / "identity-check.json").read_text(encoding="utf-8"))
        result_path = directory / "validated-review-private.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (manifest["packet_file_sha256"] != EXPECTED_PACKET_SHA
                or identity["packet_sha256"] != EXPECTED_PACKET_SHA
                or manifest["payload_sha256"] != result["payload_sha256"]
                or not identity["model_identity_verified"]
                or identity["response_model"] != "deepseek-flash"
                or result["model"] != "deepseek-flash"):
            raise ValueError("deepseek_identity_or_payload_binding_mismatch")
        for vote in result["reviews"]:
            case_id = vote["case_id"]
            if case_id in deepseek or case_id not in packet_by_id:
                raise ValueError("duplicate_or_unknown_deepseek_case")
            valid, reason = valid_vote(packet_by_id[case_id], vote)
            if not valid:
                raise ValueError(f"deepseek_vote_invalid:{case_id}:{reason}")
            two_document_citations(packet_by_id[case_id], vote)
            deepseek[case_id] = vote
        batch_hashes[name] = {"result": sha(result_path), "identity": sha(directory / "identity-check.json"),
                              "journal": sha(directory / "request-journal-private.jsonl"),
                              "raw": sha(directory / "responses-private/attempt-01-raw.json")}
    if set(deepseek) != set(packet_by_id):
        raise ValueError("missing_deepseek_votes")
    approved, excluded, vote_ledger = [], [], []
    for i, (item, packet, ancestor) in enumerate(zip(candidates, packets, ancestry["rows"])):
        case_id = f"g{i:03d}"
        if packet["case_id"] != case_id or ancestor["source_id"] != item["source_id"]:
            raise ValueError("frozen_order_binding_mismatch")
        if cpu[item["source_id"]]["status"] != "pass":
            raise ValueError("CPU_gate_failed")
        lv, dv = luna[case_id], deepseek[case_id]
        if type(lv.get("approved")) is not bool or not isinstance(lv.get("reason"), str) or not lv["reason"].strip():
            raise ValueError("luna_decision_malformed")
        two_document_citations(packet, lv, luna=True)
        reasons = []
        if ancestor["reasons"]:
            reasons.extend("ancestor:" + x for x in ancestor["reasons"])
        if not dv["approved"]:
            reasons.append("deepseek_semantic_rejection")
        if not lv["approved"]:
            reasons.append("luna_semantic_rejection")
        record = {"case_id": case_id, "source_id": item["source_id"],
                  "deepseek_approved": dv["approved"], "luna_approved": lv["approved"],
                  "cpu_passed": True, "ancestor_overlap_reasons": ancestor["reasons"],
                  "disposition": "qualified" if not reasons else "excluded", "reasons": reasons}
        vote_ledger.append(record)
        if reasons:
            excluded.append(record)
        else:
            case = make_case(item)
            case.update({"case_id": case_id, "source_split": "official HotpotQA distractor train",
                "evaluation_role": "project-isolated common-category synthetic-history evaluation",
                "supporting_facts": item["supporting_facts"],
                "cpu_check": cpu[item["source_id"]]["check"],
                "review_refs": {"deepseek_batch": next(b for b in BATCHES if
                    int(b[2:4]) <= i <= int(b[5:7])), "luna_case_id": case_id}})
            approved.append(case)
    qualified = len(approved)
    if qualified != 16 or len(excluded) != 3:
        raise ValueError(f"unexpected_frozen_subset:{qualified}:{len(excluded)}")
    asset_paths = [OUT / "PROTOCOL.md", OUT / "locator.jsonl", OUT / "screen-ledger.jsonl",
        OUT / "semantic-ledger.jsonl", OUT / "review-candidates.jsonl", OUT / "review-packets.jsonl",
        OUT / "cpu-preflight.jsonl", OUT / "cpu-preflight-v2.jsonl", OUT / "ancestry-audit.json",
        OUT / "luna-votes.json", OUT / "luna-votes-technical-correction.json",
        ROOT / "data/hotpot-full-train-v114/download-provenance.json",
        ROOT / "data/hotpot-full-train-v114/train-00000-of-00002.parquet",
        ROOT / "data/hotpot-full-train-v114/train-00001-of-00002.parquet",
        ROOT / "data/hotpot-agent-v041/train.jsonl", ROOT / "data/hotpot-agent-v041/validation.jsonl",
        ROOT / "scripts/run_v133_goal_g_cpu.py", ROOT / "scripts/audit_v133_goal_g_ancestry.py",
        ROOT / "scripts/finalize_v133_goal_g_cpu.py", ROOT / "src/mini_agent_rl/training/sft.py",
        ROOT / "scripts/v128_review_protocol.py", ROOT / "scripts/v129_review_protocol.py",
        ROOT / "scripts/v130_review_protocol.py"]
    asset_paths += [ROOT / x for x in EXPOSURE_FILES.values()]
    tokenizer_dir = Path("/root/autodl-tmp/qwen_08b")
    asset_paths += [tokenizer_dir / name for name in
        ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")]
    hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path)
              for path in asset_paths}
    writel(OUT / "vote-disposition.jsonl", vote_ledger)
    writel(OUT / "excluded-reviewed.jsonl", excluded)
    writel(OUT / "evaluation-frozen.jsonl", approved)
    manifest = {"version": "v133-goal-g-cpu-freeze-v1", "status": "qualified_partial_evaluation_set",
        "cpu_ready": True, "evaluation_isolation_verified": True, "dual_review_complete": True,
        "qualified_count": qualified, "target_count": 20, "opened_full_rows": 100,
        "unmet_conditions": ["target_shortfall_4_within_frozen_100_full_row_budget"],
        "project_exposure_scope": "v041 ancestor train/validation, v117/v118 actual train, v122 histories, v123/v130 screening, Goal C/D train/dev; base pretraining unknown",
        "gpu_used": False, "model_weights_loaded": False, "generation_or_training": False,
        "runtime_versions": {name: metadata.version(name) for name in ("torch", "transformers", "tokenizers", "pyarrow")},
        "input_sha256": hashes, "deepseek_batches_sha256": batch_hashes,
        "output_sha256": {name: sha(OUT / name) for name in
            ("vote-disposition.jsonl", "excluded-reviewed.jsonl", "evaluation-frozen.jsonl")}}
    write(OUT / "freeze-manifest.json", manifest)
    print(json.dumps({"qualified": qualified, "reviewed": len(vote_ledger), "excluded": len(excluded),
                      "manifest_sha256": sha(OUT / "freeze-manifest.json")}))


if __name__ == "__main__":
    main()
