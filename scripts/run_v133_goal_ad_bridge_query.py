"""Goal AD: diagnose evidence gaps and train correction queries at real failures."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import run_v133_goal_ab_evidence_decision as ab
import run_v133_goal_aa_audit_confirmation as aa
import run_v133_goal_ac_failure_state_repair as ac
import run_v133_goal_q_search as q
import run_v133_goal_z_multihop_sft as z
from mini_agent_rl.model.context import _payload

ROOT = z.ROOT
OUT = ROOT / "runs/v133-goal-ad-bridge-query"
CFG, STATE = OUT / "config.json", OUT / "state.json"
SOURCE = z.OUT / "seed42-trajectory/adapter-final"
SPLITS = ("dev", "old", "y", "aa_regression")
MIN_PAIRS, MAX_PAIRS, MIN_BRIDGE = 32, 48, 16
MIN_REPAIRABLE_CASES = 8
TOOL_RESPONSE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.DOTALL)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    value = json.loads(text) if text.lstrip().startswith("[") else None
    return value if isinstance(value, list) else [json.loads(x) for x in text.splitlines() if x.strip()]


def write(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()


def append(path: Path, value: dict) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def set_state(status: str, **updates) -> dict:
    value = read(STATE)
    value.update(status=status, **updates)
    write(STATE, value)
    return value


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(q.p.BASE, local_files_only=True)


def decode_prompt(token_ids: list[int], tokenizer) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def tool_payloads(actual_prompt: str) -> list[dict]:
    """Read only tool responses physically present in the saved prompt tokens."""
    found = []
    for match in TOOL_RESPONSE.finditer(actual_prompt):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            found.append({"payload": payload, "char_start": match.start(1), "char_end": match.end(1)})
    return found


def visible_documents(actual_prompt: str) -> list[dict]:
    """Return exact document text and input offsets from actual, rendered tool messages."""
    docs, seen = [], set()
    for tool_index, item in enumerate(tool_payloads(actual_prompt)):
        payload = item["payload"]
        for rank, doc in enumerate(payload.get("result") or [], 1):
            if not isinstance(doc, dict) or not doc.get("id"):
                continue
            doc_id = str(doc["id"])
            key = (doc_id, str(doc.get("text", "")))
            if key in seen:
                continue
            seen.add(key)
            raw_text = str(doc.get("text", ""))
            encoded_text = json.dumps(raw_text, ensure_ascii=False)[1:-1]
            text_start = actual_prompt.find(encoded_text, item["char_start"], item["char_end"]) if encoded_text else -1
            docs.append({"id": doc_id, "title": str(doc.get("title", "")),
                "text": raw_text, "score": doc.get("score"),
                "tool_index": tool_index, "rank": rank,
                "prompt_char_span": [item["char_start"], item["char_end"]],
                "text_char_span": [text_start, text_start + len(encoded_text)] if text_start >= 0 else None,
                "fragment_complete": True})
    return docs


def sources() -> dict[str, Path]:
    return {"adapter": SOURCE / "adapter_model.safetensors",
        "tokenizer": q.p.BASE / "tokenizer.json",
        "base_weights": sorted(q.p.BASE.glob("*.safetensors"))[0],
        "z_config": z.CFG, "z_train_inputs": z.OUT / "training-inputs.jsonl",
        "z_train_cases": z.OUT / "selected-train.jsonl",
        "z_train_corpus": z.OUT / "corpus-train.json",
        "z_dev_cases": z.OUT / "selected-dev.jsonl", "z_dev_corpus": z.OUT / "corpus-dev.json",
        "z_confirm_cases": z.OUT / "selected-confirm.jsonl", "z_confirm_corpus": z.OUT / "corpus-confirm.json",
        "ac_state": ac.STATE, "ac_source_paths": ac.OUT / "selected-repair-paths.jsonl",
        "ac_evidence_final": ac.OUT / "repair-evidence-final.json",
        "ac_material_decision": ac.OUT / "repair-material-decision.json",
        "ab_review": ab.OUT / "development-review-final.jsonl",
        "ab_review_map": ab.OUT / "development-review-map.json",
        "ab_confirmation_manifest": ab.OUT / "fresh-confirm-manifest.json",
        "aa_confirm_cases": aa.OUT / "confirmation-revised-results.jsonl",
        "model_code": ROOT / "src/mini_agent_rl/model/local.py",
        "context_code": ROOT / "src/mini_agent_rl/model/context.py",
        "search_code": ROOT / "src/mini_agent_rl/tools/search.py",
        "ad_runner": Path(__file__)}


def software() -> dict:
    import torch, transformers, peft
    return {"python": __import__("sys").version.split()[0], "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "env": {k: __import__("os").environ.get(k) for k in
            ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")}}


def archived_paths() -> list[dict]:
    values = rows(ac.OUT / "selected-repair-paths.jsonl")
    if len(values) != 48 or len({x["case_id"] for x in values}) != 48:
        raise RuntimeError("AC path archive must contain its frozen 48 candidates")
    return values


def repair_corpus_key(case_id: str, goal_z_training_ids: set[str]) -> str:
    """Corpus ownership follows exact frozen case membership, not source label."""
    return "goal_z_train" if case_id in goal_z_training_ids else "ac_extra_train"


def audit_packet() -> dict:
    tok = load_tokenizer()
    path_by_id = {x["case_id"]: x for x in archived_paths()}
    previous_votes = {x["case_id"]: x for x in ac.rows(ac.OUT / "repair-evidence-final.json")}
    packet = []
    for case_id, path in path_by_id.items():
        final_action = path["repair_actions"][-1]
        actual = decode_prompt(final_action["prompt_token_ids"], tok)
        if path.get("actual_final_prompt") != actual:
            raise RuntimeError(f"Saved final prompt string differs from its token replay: {case_id}")
        docs = visible_documents(actual)
        steps = []
        for index, action in enumerate(path["repair_actions"]):
            prompt = decode_prompt(action["prompt_token_ids"], tok)
            step_docs = visible_documents(prompt)
            steps.append({"index": index, "action": action["action"], "query": action.get("query"),
                "answer": action.get("answer"), "prompt_sha256": hashlib.sha256(
                    json.dumps(action["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest(),
                "actual_prompt": prompt, "visible_documents": step_docs})
        prior = previous_votes[case_id]
        insufficient = path["insufficient_state"]
        insufficient_prompt = decode_prompt(insufficient["prompt_token_ids"], tok)
        insufficient_docs = visible_documents(insufficient_prompt)
        packet.append({"case_id": case_id, "source": path["source"], "question": path["question"],
            "target_answer": path["training_answer"], "original_event": path["event"],
            "prior_audit_decision": prior["approved"], "prior_reasons": [
                prior["reviewer_a"].get("reason", ""), prior["reviewer_b"].get("reason", "")],
            "actual_insufficient_prompt": insufficient_prompt,
            "insufficient_prompt_sha256": hashlib.sha256(json.dumps(
                insufficient["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest(),
            "insufficient_visible_documents": insufficient_docs,
            "original_query_at_failure": insufficient.get("query"),
            "actual_final_prompt": actual, "steps": steps, "visible_documents": docs,
            "required_citations": prior["reviewer_a"].get("supporting_snippets", []) +
                prior["reviewer_b"].get("supporting_snippets", []),
            "instructions": "Use actual_prompt and visible_documents only to judge what was visible. Text fragments may be truncated. The target answer is shown only for evidence validation. Cite exact visible substrings and char offsets."})
    packet.sort(key=lambda x: hashlib.sha256(("goal-ad-path:" + x["case_id"]).encode()).hexdigest())
    write_jsonl(OUT / "path-review-packet.jsonl", packet)
    return {"items": len(packet), "sha256": sha(OUT / "path-review-packet.jsonl")}


def prepare() -> dict:
    # Recover only a preflight interrupted after immutable config was written but
    # before the first audit packet/state could be saved.
    if OUT.exists() and CFG.exists() and not STATE.exists():
        cfg = read(CFG)
        previous_hashes = cfg["source_hashes"]
        current_hashes = {k: sha(v) for k, v in sources().items()}
        changed = {k for k in previous_hashes if previous_hashes[k] != current_hashes.get(k)}
        if changed - {"ad_runner"}:
            raise RuntimeError(f"Partial prepare recovery found changed frozen inputs: {sorted(changed)}")
        write(OUT / "config-preflight-failed-v1.json", cfg)
        cfg["source_hashes"] = current_hashes
        cfg["preflight_recovery"] = {"prior_runner_sha256": previous_hashes.get("ad_runner"),
            "recovery_runner_sha256": current_hashes["ad_runner"],
            "prior_attempt_error": "evidence parser incorrectly required a nonempty tool result"}
        write(CFG, cfg)
        cfg = verify()
        packet = audit_packet()
        write(STATE, {"status": "awaiting_path_adjudication", "confirmation_accessed": False,
            "stages": [{"name": "AD0_prepare_recovered", "seconds": time.time() - cfg["created_at"]}],
            "path_review_packet": packet, "gpu_seconds": 0.,
            "preflight_deviation": "Schema-only inspection of the sealed confirmation file had occurred before prepare; no records or answers were emitted or used. See config for details."})
        return {"status": "awaiting_path_adjudication", "packet": packet,
            "recovered_partial_prepare": True, "confirmation_accessed": False}
    if OUT.exists() or CFG.exists() or STATE.exists():
        raise FileExistsError(f"Refusing to overwrite Goal AD run: {OUT}")
    for key, path in sources().items():
        if key == "base_weights":
            if not path.is_file(): raise FileNotFoundError(path)
        elif not Path(path).is_file():
            raise FileNotFoundError(f"Missing frozen source {key}: {path}")
    if read(ac.STATE)["status"] != "stopped_insufficient_repair_evidence":
        raise RuntimeError("Goal AC must be terminalized at its frozen evidence gate")
    ac_config = ac.verify()
    manifest = read(ab.OUT / "fresh-confirm-manifest.json")
    if manifest.get("confirmation_accessed"):
        raise RuntimeError("The one-time fresh confirmation was already accessed")
    if len(rows(z.OUT / "selected-train.jsonl")) != 128:
        raise RuntimeError("Goal Z training set changed")
    sw = software()
    if not sw["gpu"]:
        raise RuntimeError("AutoDL GPU is unavailable")
    hashes = {k: sha(v) for k, v in sources().items()}
    archived_paths()
    baseline = ac.frozen_ab_scores()[0]
    order = {seed: list(z.read(ab.OUT / "config.json")["order"][str(seed)]) for seed in (42, 43)}
    OUT.mkdir(parents=True)
    config = {"version": "goal-ad-bridge-query-v1", "source_hashes": hashes,
        "software": sw, "product": z.read(z.CFG)["product"], "start_adapter": str(SOURCE),
        "ac_run_sha256": sha(ac.CFG), "order": {str(k): v for k, v in order.items()},
        "seeds": [42, 43], "updates_per_model": 128, "arms": ["A", "B"],
        "material_gate": {"min_pairs": MIN_PAIRS, "max_pairs": MAX_PAIRS,
            "min_bridge_repairs": MIN_BRIDGE, "min_repairable_diagnostic_cases": MIN_REPAIRABLE_CASES},
        "optimization": {"learning_rate": 5e-6, "betas": [.9, .999], "eps": 1e-8,
            "weight_decay": 0., "gradient_clip": 1.},
        "baseline_scores": baseline, "confirmation_access_note": {
            "schema_only_deserialization_before_prepare": True,
            "command_output_contained_only": ["record_count", "field_names"],
            "no_confirm_outputs_generated_or_used_for_selection": True},
        "confirmation_sha256": {
            "cases": sha(ab.OUT / "fresh-confirm-cases.jsonl"),
            "corpus": sha(ab.OUT / "fresh-confirm-corpus.json")},
        "created_at": time.time()}
    write(CFG, config)
    packet = audit_packet()
    write(STATE, {"status": "awaiting_path_adjudication", "confirmation_accessed": False,
        "stages": [{"name": "AD0_prepare", "seconds": time.time() - config["created_at"]}],
        "path_review_packet": packet, "gpu_seconds": 0.})
    return {"status": "awaiting_path_adjudication", "packet": packet, "source_hashes": hashes,
        "baseline_dev": baseline.get("dev-baseline"), "confirmation_accessed": False}


def verify() -> dict:
    cfg = read(CFG)
    if {k: sha(v) for k, v in sources().items()} != cfg["source_hashes"]:
        raise RuntimeError("Frozen Goal AD inputs changed")
    if software() != cfg["software"]:
        raise RuntimeError("GPU or software environment changed")
    if sha(ab.OUT / "fresh-confirm-cases.jsonl") != cfg["confirmation_sha256"]["cases"] or \
       sha(ab.OUT / "fresh-confirm-corpus.json") != cfg["confirmation_sha256"]["corpus"] or \
       read(ab.OUT / "fresh-confirm-manifest.json").get("confirmation_accessed"):
        raise RuntimeError("Fresh confirmation state or hash changed before authorization")
    return cfg


def refresh_pretraining_runner_hash() -> None:
    """Freeze the final runner revision before any audit decision or training starts."""
    if read(STATE)["status"] != "awaiting_path_adjudication":
        raise RuntimeError("Runner hash may only be refreshed before path adjudication")
    cfg = read(CFG)
    current_hashes = {k: sha(v) for k, v in sources().items()}
    changed = {k for k in cfg["source_hashes"] if cfg["source_hashes"][k] != current_hashes.get(k)}
    # Once the reviewed code hash is already frozen, submitted review files are
    # expected and must not block an idempotent run() call.
    if not changed:
        return
    if any((OUT / name).exists() for name in ("root-path-adjudication.jsonl", "failure-diagnosis.jsonl",
            "query-proposals.jsonl", "repair-review-a.jsonl", "repair-review-b.jsonl")):
        raise RuntimeError("Cannot change frozen runner after external audit input was submitted")
    if changed - {"ad_runner"}:
        raise RuntimeError(f"AD frozen assets changed before review: {sorted(changed)}")
    archive = OUT / f"config-runner-revision-{len(cfg.get('runner_revision_history', []))+1:02d}.json"
    if archive.exists(): raise FileExistsError(archive)
    write(archive, cfg)
    cfg.setdefault("runner_revision_history", []).append({"previous_sha256": cfg["source_hashes"]["ad_runner"],
        "current_sha256": current_hashes["ad_runner"],
        "reason": "Completed evaluation packet, scoring, and guard implementation before audit submission."})
    cfg["source_hashes"] = current_hashes
    cfg["confirmation_access_note"] = {
        "schema_only_deserialization_before_prepare": True,
        "command_output_contained_only": ["record_count", "field_names"],
        "no_confirm_outputs_generated_or_used_for_selection": True,
        "protocol_deviation": "The file was parsed to inspect its schema before the dev gate; no question or answer values were emitted to the agent output or used for model decisions."}
    write(CFG, cfg)


def check_id_rows(packet: list[dict], votes: list[dict], key="case_id") -> dict[str, dict]:
    expected = [x[key] for x in packet]
    actual = [x.get(key) for x in votes]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise RuntimeError(f"Review IDs must cover the frozen packet exactly: {key}")
    return {x[key]: x for x in votes}


def path_adjudication() -> dict:
    packet = rows(OUT / "path-review-packet.jsonl")
    votes = rows(OUT / "root-path-adjudication.jsonl")
    if len(packet) != 48:
        raise RuntimeError("AD0 path review packet is incomplete")
    by_id = check_id_rows(packet, votes)
    allowed = {"supported", "unsupported", "ambiguous"}
    for item in packet:
        vote = by_id[item["case_id"]]
        if vote.get("verdict") not in allowed or not vote.get("reason"):
            raise RuntimeError(f"Incomplete primary adjudication: {item['case_id']}")
        if vote["verdict"] == "supported":
            ids = set(vote.get("evidence_ids", []))
            docs = {x["id"]: x for x in item["visible_documents"]}
            citations = vote.get("citations", [])
            if not ids or not ids.issubset(docs) or not citations:
                raise RuntimeError("Supported path lacks visible cited evidence")
            for citation in citations:
                doc = docs.get(citation.get("document_id"))
                if not doc or not citation.get("text") or citation["text"] not in doc["text"]:
                    raise RuntimeError("Primary citation is not an exact visible substring")
        if vote.get("bridge_sources_valid") is not True and vote["verdict"] == "supported":
            raise RuntimeError("A supported query path requires verified bridge sources")
    result = {"cases": len(packet), "supported": sum(x["verdict"] == "supported" for x in votes),
        "unsupported": sum(x["verdict"] == "unsupported" for x in votes),
        "ambiguous": sum(x["verdict"] == "ambiguous" for x in votes)}
    write(OUT / "path-adjudication-summary.json", result)
    diagnostics = rows(OUT / "failure-diagnosis.jsonl")
    failed = [x for x in packet if by_id[x["case_id"]]["verdict"] != "supported"]
    diag_by_id = check_id_rows(failed, diagnostics)
    allowed_causes = {"missing_from_corpus", "wrong_query", "truncated_or_missing_in_input",
        "budget_insufficient", "label_or_review_issue"}
    for item in failed:
        vote = diag_by_id[item["case_id"]]
        if vote.get("primary_cause") not in allowed_causes or not vote.get("reason"):
            raise RuntimeError(f"Incomplete failure diagnosis: {item['case_id']}")
        if vote.get("query_remediable") and vote["primary_cause"] != "wrong_query":
            raise RuntimeError("Only wrong-query cases count as query-remediable")
    count = len({x["case_id"] for x in diagnostics if x.get("query_remediable") is True})
    gate = {"path_review": result, "diagnosed_failures": len(failed),
        "query_remediable_cases": count, "passed": count >= MIN_REPAIRABLE_CASES}
    write(OUT / "diagnostic-gate.json", gate)
    if not gate["passed"]:
        set_state("complete_diagnostic_gate_failed", diagnostic_gate=gate)
    else:
        write_query_author_packet(failed, diag_by_id)
        set_state("awaiting_query_proposals", diagnostic_gate=gate)
    return gate


def validate_query_proposals() -> dict:
    packet = rows(OUT / "query-author-packet.jsonl")
    proposals = rows(OUT / "query-proposals.jsonl")
    paths = {x["case_id"]: x for x in archived_paths()}
    diagnostics = {x["case_id"]: x for x in rows(OUT / "failure-diagnosis.jsonl")}
    selected, result = validate_query_proposal_rows(packet, proposals, paths, diagnostics)
    write_jsonl(OUT / "validated-query-proposals.jsonl", selected)
    write(OUT / "query-proposal-gate.json", result)
    if not result["passed"]:
        set_state("complete_query_material_gate_failed", query_gate=result)
    else:
        set_state("building_repair_paths", query_gate=result)
    return result


def validate_query_proposal_rows(packet: list[dict], proposals: list[dict],
        paths: dict[str, dict], diagnostics: dict[str, dict]) -> tuple[list[dict], dict]:
    by_id = check_id_rows(packet, proposals)
    eligible = {k for k, x in diagnostics.items() if x["query_remediable"]}
    selected = []
    for item in packet:
        proposal = by_id[item["case_id"]]
        if item["case_id"] not in eligible:
            if proposal.get("queries"):
                raise RuntimeError("Query proposal supplied for a non-remediable case")
            continue
        if len(proposal.get("queries", [])) != 1:
            raise RuntimeError("Each remediable case requires exactly one bounded repair query")
        visible_before = "\n".join([item["question"], item["actual_insufficient_prompt"]])
        if proposal.get("input_sha256") != hashlib.sha256(visible_before.encode()).hexdigest():
            raise RuntimeError("Query author did not use the frozen visible-only input")
        searches_used = int(item["searches_before_error"])
        if searches_used + len(proposal["queries"]) > 3:
            raise RuntimeError("Proposed correction exceeds search budget")
        target = paths[item["case_id"]]["training_answer"]
        for query_index, spec in enumerate(proposal["queries"]):
            query = spec.get("query", "")
            source_text = spec.get("bridge_source", "")
            if not isinstance(query, str) or not query.strip() or not isinstance(source_text, str) or not source_text:
                raise RuntimeError("Query proposal lacks text or bridge source")
            if source_text.casefold() not in visible_before.casefold() and query_index == 0:
                raise RuntimeError("First query bridge entity is not grounded in actual visible input")
            if source_text.casefold() not in query.casefold():
                raise RuntimeError("Proposed query does not use its cited bridge entity")
            target_norm, query_norm = ac.source.norm(target), ac.source.norm(query)
            if len(target_norm) >= 5 and target_norm in query_norm and target_norm not in ac.source.norm(visible_before):
                raise RuntimeError("Query contains an answer string absent from visible input")
            selected.append({"case_id": item["case_id"], "query_index": query_index,
                "query": query.strip(), "bridge_source": source_text})
    result = {"proposals": len(selected), "eligible_cases": len(eligible),
        "remediable_cases_with_path": len({x["case_id"] for x in selected}),
        "passed": len({x["case_id"] for x in selected}) >= MIN_REPAIRABLE_CASES}
    return selected, result


def execute_repair_queries() -> dict:
    """Execute each frozen query once through the same Goal Q tool implementation."""
    if read(STATE)["status"] != "building_repair_paths":
        raise RuntimeError("Repair query construction is not authorized by the gate")
    cfg = verify()
    proposals = rows(OUT / "validated-query-proposals.jsonl")
    paths = {x["case_id"]: x for x in archived_paths()}
    from mini_agent_rl.domain import Message
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    from transformers import AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(q.p.BASE, local_files_only=True)
    output = OUT / "repair-search-executions.jsonl"
    if output.exists():
        raise FileExistsError(output)
    groups: dict[str, list[dict]] = {}
    for row in proposals: groups.setdefault(row["case_id"], []).append(row)
    goal_z_training_ids = {row["id"] for row in z.cases("train")}
    for case_id, specs in groups.items():
        source = paths[case_id]
        insufficient = source["insufficient_state"]
        prompt_ids = list(insufficient["prompt_token_ids"])
        prompt = tok.decode(prompt_ids, skip_special_tokens=False)
        corpus_key = repair_corpus_key(case_id, goal_z_training_ids)
        corpus_path = z.OUT / "corpus-train.json" if corpus_key == "goal_z_train" else ac.OUT / "corpus-extra-train.json"
        client, _, registry = z.make_client(SOURCE, corpus_path)
        history = parse_prompt_messages(prompt, Message)
        packed = client._inputs(history, "goal_ad_exact_prefix")
        if packed["input_ids"][0].tolist() != prompt_ids or packed["attention_mask"][0].tolist() != insufficient["prompt_attention_mask"]:
            raise RuntimeError(f"Saved actual failure prefix does not replay exactly: {case_id}")
        prior_queries = assistant_queries(history)
        if len(prior_queries) + len(specs) > 4 or sum(1 for x in history if x.role == "tool") + len(specs) > 3:
            raise RuntimeError("Frozen correction would exceed product call/search budget")
        results = []
        for spec in specs:
            key = " ".join(spec["query"].casefold().split())
            if key in {" ".join(x.casefold().split()) for x in prior_queries}:
                raise RuntimeError("Corrective query duplicates a query already executed")
            observation = await_execute(registry, spec["query"])
            if not observation.get("ok"):
                raise RuntimeError(f"Repair search failed; preserving the one-attempt stop: {case_id}")
            seen = set(" ".join(x.casefold().split()) for x in prior_queries)
            seen.add(key)
            observation["searched_queries"] = sorted(seen)
            tool_content = json.dumps({k: v for k, v in observation.items() if k != "latency_ms"}, ensure_ascii=False)
            raw_action, action_ids = exact_search_tokens(tok, spec["query"])
            before = client._inputs(history, "goal_ad_repair_query")
            action_prompt_ids = before["input_ids"][0].tolist()
            action_prompt_mask = before["attention_mask"][0].tolist()
            history.extend([Message(role="assistant", content=raw_action), Message(role="tool", content=tool_content)])
            after = client._inputs(history, "goal_ad_post_repair_evidence")
            after_ids, after_mask = after["input_ids"][0].tolist(), after["attention_mask"][0].tolist()
            after_prompt = tok.decode(after_ids, skip_special_tokens=False)
            returned = visible_documents(after_prompt)
            results.append({"query": spec["query"], "bridge_source": spec["bridge_source"],
                "action_token_ids": action_ids, "action_raw": raw_action,
                "prompt_token_ids": action_prompt_ids, "prompt_attention_mask": action_prompt_mask,
                "observation": observation, "visible_documents_after": returned,
                "after_prompt_token_ids": after_ids, "after_prompt_attention_mask": after_mask,
                "after_prompt_sha256": hashlib.sha256(json.dumps(after_ids, separators=(",", ":")).encode()).hexdigest()})
            prior_queries.append(spec["query"])
        result = {"case_id": case_id, "source": source["source"], "event": source["event"],
            "question": source["question"], "target_answer": source["training_answer"],
            "searches": results, "insufficient_prompt_token_ids": prompt_ids,
            "insufficient_prompt_attention_mask": insufficient["prompt_attention_mask"],
            "sufficient_prompt_token_ids": results[-1]["after_prompt_token_ids"],
            "sufficient_prompt_attention_mask": results[-1]["after_prompt_attention_mask"],
            "sufficient_prompt_sha256": results[-1]["after_prompt_sha256"]}
        append(output, result)
        print(json.dumps({"case_id": case_id, "queries": len(results), "status": "tools_executed"}), flush=True)
        del client, registry
        gc.collect(); torch.cuda.empty_cache()
    packet = build_repair_review_packet()
    return {"cases": len(groups), "queries": len(proposals), "sha256": sha(output),
        "evidence_review": packet}


def await_execute(registry, query: str) -> dict:
    return asyncio.run(registry.execute("search", {"query": query}))


def exact_search_tokens(tokenizer, query: str) -> tuple[str, list[int]]:
    raw = json.dumps({"action": "search", "query": query}, ensure_ascii=False)
    ids = tokenizer.encode(raw, add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(end_id, int) or end_id < 0:
        raise RuntimeError("Tokenizer lacks the product message terminator")
    return raw, ids + [end_id]


def parse_prompt_messages(prompt: str, Message) -> list:
    """Recover message history from saved Qwen chat tokens for packer replay."""
    pattern = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)(?:<\|im_end\|>|(?=<\|im_start\|>))", re.DOTALL)
    matches = list(pattern.finditer(prompt))
    if not matches:
        raise ValueError("Cannot parse saved chat prompt")
    messages = []
    for match in matches:
        role, content = match.group(1), match.group(2)
        if role == "assistant" and "<think>" in content and content.rstrip().endswith("</think>"):
            continue  # final generation prefix comes from apply_chat_template
        if role == "user" and content.startswith("<tool_response>") and content.endswith("</tool_response>"):
            content = content[len("<tool_response>"): -len("</tool_response>")].strip()
            role = "tool"
        messages.append(Message(role=role, content=content))
    if not messages or messages[-1].role == "assistant":
        messages = [x for x in messages if x.role != "assistant"]
    return messages


def assistant_queries(messages: list) -> list[str]:
    found = []
    for item in messages:
        if item.role != "assistant": continue
        try: value = json.loads(item.content)
        except (TypeError, json.JSONDecodeError): continue
        if value.get("action") == "search" and isinstance(value.get("query"), str):
            found.append(value["query"])
    return found


def build_repair_review_packet() -> dict:
    tok = load_tokenizer()
    path_votes = {x["case_id"]: x for x in rows(OUT / "root-path-adjudication.jsonl")}
    archived = {x["case_id"]: x for x in archived_paths()}
    candidates = []
    for case_id, vote in path_votes.items():
        if vote["verdict"] != "supported": continue
        path = archived[case_id]
        actions = [x for x in path["repair_actions"] if x["action"] == "search"]
        if not actions: continue
        sufficient = decode_prompt(path["sufficient_state"]["prompt_token_ids"], tok)
        candidates.append({"case_id": case_id, "original_event": path["event"],
            "source": path["source"], "question": path["question"],
            "target_answer": path["training_answer"],
            "search_actions": [{"query": x["query"], "bridge_source": x.get("query_source", ""),
                "prompt_token_ids": x["prompt_token_ids"], "prompt_attention_mask": x["prompt_attention_mask"],
                "action_token_ids": x["action_token_ids"]} for x in actions],
            "sufficient_prompt_token_ids": path["sufficient_state"]["prompt_token_ids"],
            "sufficient_prompt_attention_mask": path["sufficient_state"]["prompt_attention_mask"],
            "sufficient_prompt": sufficient, "visible_documents": visible_documents(sufficient),
            "material_origin": "reused_AC_path"})
    for item in rows(OUT / "repair-search-executions.jsonl"):
        final = item["searches"][-1]
        prompt = decode_prompt(item["sufficient_prompt_token_ids"], tok)
        candidates.append({"case_id": item["case_id"], "original_event": item["event"],
            "source": item["source"], "question": item["question"],
            "target_answer": item["target_answer"],
            "search_actions": [{"query": x["query"], "bridge_source": x["bridge_source"],
                "prompt_token_ids": x["prompt_token_ids"], "prompt_attention_mask": x["prompt_attention_mask"],
                "action_token_ids": x["action_token_ids"]} for x in item["searches"]],
            "sufficient_prompt_token_ids": item["sufficient_prompt_token_ids"],
            "sufficient_prompt_attention_mask": item["sufficient_prompt_attention_mask"],
            "sufficient_prompt": prompt, "visible_documents": visible_documents(prompt),
            "material_origin": "new_real_failure_query"})
    if len({x["case_id"] for x in candidates}) != len(candidates):
        raise RuntimeError("Duplicate repaired training case")
    candidates.sort(key=lambda x: (x["original_event"] != "wrong_bridge_query",
        hashlib.sha256(("goal-ad-evidence:" + x["case_id"]).encode()).hexdigest()))
    candidates = candidates[:MAX_PAIRS]
    write_jsonl(OUT / "repair-training-candidates.jsonl", candidates)
    packet = []
    for item in candidates:
        packet.append({"case_id": item["case_id"], "question": item["question"],
            "target_answer": item["target_answer"], "original_event": item["original_event"],
            "material_origin": item["material_origin"],
            "search_actions": [{"query": x["query"], "bridge_source": x["bridge_source"]}
                for x in item["search_actions"]],
            "search_action_prompts": [decode_prompt(x["prompt_token_ids"], tok) for x in item["search_actions"]],
            "sufficient_state": {"actual_prompt": item["sufficient_prompt"],
                "visible_documents": item["visible_documents"]},
            "instructions": "Verify every bridge source was visible before its search action. Decide whether the final answer is fully supported by passages in the sufficient-state actual prompt. Quote exact substrings from visible_documents."})
    write_jsonl(OUT / "repair-review-packet.jsonl", packet)
    set_state("awaiting_repair_evidence_review", repair_review={"candidates": len(packet),
        "reused_AC_paths": sum(x["material_origin"] == "reused_AC_path" for x in candidates),
        "new_failure_paths": sum(x["material_origin"] != "reused_AC_path" for x in candidates)})
    return {"candidates": len(packet), "sha256": sha(OUT / "repair-review-packet.jsonl")}


def run() -> dict:
    if not CFG.exists():
        raise FileNotFoundError("Run prepare first")
    if read(STATE)["status"] == "awaiting_path_adjudication":
        refresh_pretraining_runner_hash()
    verify()
    status = read(STATE)["status"]
    if status == "awaiting_path_adjudication":
        if not (OUT / "root-path-adjudication.jsonl").exists() or not (OUT / "failure-diagnosis.jsonl").exists():
            return {"status": status, "awaiting": ["root-path-adjudication.jsonl", "failure-diagnosis.jsonl"]}
        return path_adjudication()
    if status == "awaiting_query_proposals":
        if not (OUT / "query-proposals.jsonl").exists():
            return {"status": status, "awaiting": ["query-proposals.jsonl"]}
        validate_query_proposals()
        return execute_repair_queries()
    if status == "building_repair_paths":
        return execute_repair_queries()
    if status == "awaiting_repair_evidence_review":
        a, b = OUT / "repair-review-a.jsonl", OUT / "repair-review-b.jsonl"
        if not a.exists() or not b.exists():
            return {"status": status, "awaiting": [str(x.name) for x in (a, b) if not x.exists()]}
        return finalize_repairs()
    if status == "ready_to_train":
        return run_training()
    if status == "training_complete":
        return evaluate_development()
    if status == "awaiting_development_review":
        if not (OUT / "development-review-final.jsonl").exists():
            return {"status": status, "awaiting": ["development-review-final.jsonl"]}
        return development_decision()
    if status == "confirmation_authorized":
        return evaluate_confirmation()
    if status == "awaiting_confirmation_review":
        if not (OUT / "confirmation-review-final.jsonl").exists():
            return {"status": status, "awaiting": ["confirmation-review-final.jsonl"]}
        return confirmation_decision()
    if status in {"complete_diagnostic_gate_failed", "complete_query_material_gate_failed",
                  "complete_repair_material_gate_failed", "complete_development_gate_failed",
                  "complete_confirmation_gate_failed", "complete_confirmed_and_published"}:
        return summarize()
    raise NotImplementedError(f"Goal AD stage not yet implemented: {status}")


def write_query_author_packet(failed: list[dict], diagnosis: dict[str, dict]) -> dict:
    from mini_agent_rl.domain import Message
    packet = []
    for path in failed:
        case_id = path["case_id"]
        vote = diagnosis[case_id]
        if not vote.get("query_remediable"): continue
        prompt = path["actual_insufficient_prompt"]
        visible_only = path["question"] + "\n" + prompt
        query_count = len(assistant_queries(parse_prompt_messages(prompt, Message)))
        packet.append({"case_id": case_id, "question": path["question"],
            "actual_insufficient_prompt": prompt,
            "visible_document_ids": [x["id"] for x in path["insufficient_visible_documents"]],
            "failure_cause": vote["primary_cause"],
            "searches_before_error": query_count,
            "remaining_searches": 3 - query_count,
            "input_sha256": hashlib.sha256(visible_only.encode()).hexdigest(),
            "instructions": "Use only question and actual_insufficient_prompt. Write exactly one search query for the missing relation. Any entity in the query must appear in this input. Do not add answers, known support titles, or hidden mappings."})
    packet.sort(key=lambda x: hashlib.sha256(("goal-ad-query:" + x["case_id"]).encode()).hexdigest())
    write_jsonl(OUT / "query-author-packet.jsonl", packet)
    return {"cases": len(packet), "sha256": sha(OUT / "query-author-packet.jsonl")}


def finalize_repairs() -> dict:
    packet = rows(OUT / "repair-review-packet.jsonl")
    reviews = [rows(OUT / "repair-review-a.jsonl"), rows(OUT / "repair-review-b.jsonl")]
    bys = [check_id_rows(packet, vote) for vote in reviews]
    path_by_id = {x["case_id"]: x for x in rows(OUT / "repair-training-candidates.jsonl")}
    approved, audit = [], []
    for item in packet:
        va, vb = (table[item["case_id"]] for table in bys)
        docs = {x["id"]: x for x in item["sufficient_state"]["visible_documents"]}
        provenance = True
        candidate = path_by_id[item["case_id"]]
        for action, prompt in zip(candidate["search_actions"], item["search_action_prompts"]):
            source = str(action.get("bridge_source", ""))
            query = str(action.get("query", ""))
            if not source or source.casefold() not in prompt.casefold() or source.casefold() not in query.casefold():
                provenance = False
                break
        def citations_valid(vote):
            citations = vote.get("citations", [])
            if not citations or not vote.get("evidence_ids"):
                return False
            if not set(vote["evidence_ids"]).issubset(docs): return False
            return all(c.get("document_id") in docs and c.get("text") in docs[c["document_id"]]["text"]
                for c in citations)
        ok = (va.get("supported") is True and vb.get("supported") is True and
              va.get("bridge_sources_valid") is True and vb.get("bridge_sources_valid") is True and
              va.get("answers_relation") is True and vb.get("answers_relation") is True and
              provenance and citations_valid(va) and citations_valid(vb))
        audit.append({"case_id": item["case_id"], "approved": ok, "reviewer_a": va,
            "reviewer_b": vb, "reason": "both verified visible evidence and bridge provenance" if ok else
            "one or both reviews did not verify full relation, citation, or bridge provenance"})
        if ok: approved.append(path_by_id[item["case_id"]])
    if len(audit) != len(packet): raise RuntimeError("Repair audit incomplete")
    write(OUT / "repair-adjudication.json", audit)
    bridge = [x for x in approved if x["original_event"] == "wrong_bridge_query"]
    selected = bridge[:MAX_PAIRS]
    selected_ids = {x["case_id"] for x in selected}
    remaining = [x for x in approved if x["case_id"] not in selected_ids]
    selected.extend(remaining[:max(0, MAX_PAIRS-len(selected))])
    gate = {"reviewed": len(audit), "approved": len(approved), "selected": len(selected),
        "bridge_query_repairs": sum(x["original_event"] == "wrong_bridge_query" for x in selected),
        "passed": len(selected) >= MIN_PAIRS and sum(x["original_event"] == "wrong_bridge_query" for x in selected) >= MIN_BRIDGE}
    write_jsonl(OUT / "repair-pairs.jsonl", selected)
    write(OUT / "repair-material-decision.json", gate)
    if not gate["passed"]:
        set_state("complete_repair_material_gate_failed", repair_gate=gate)
    else:
        binding = freeze_binding(selected)
        set_state("ready_to_train", repair_gate=gate, binding=binding)
    return gate


def freeze_binding(selected: list[dict]) -> dict:
    cfg = verify()
    train_ids = set(cfg["order"]["42"])
    originals = [x for x in selected if x["case_id"] in train_ids]
    extras = sorted((x for x in selected if x["case_id"] not in train_ids),
        key=lambda x: hashlib.sha256(("goal-ad-extra-slot:"+x["case_id"]).encode()).hexdigest())
    free = sorted(train_ids - {x["case_id"] for x in originals},
        key=lambda x: hashlib.sha256(("goal-ad-free-slot:"+x).encode()).hexdigest())
    if len(extras) > len(free): raise RuntimeError("More repair paths than frozen update slots")
    mapping = {x["case_id"]: x["case_id"] for x in originals}
    mapping.update({slot: item["case_id"] for slot, item in zip(free, extras)})
    binding = {seed: [{"update": i, "case_id": case_id, "repair_case_id": mapping[case_id]}
        for i, case_id in enumerate(cfg["order"][seed], 1) if case_id in mapping]
        for seed in ("42", "43")}
    if {x["case_id"] for x in binding["42"]} != {x["case_id"] for x in binding["43"]}:
        raise RuntimeError("Matched repair bindings differ by seed")
    write(OUT / "repair-update-binding.json", binding)
    return binding


def train(seed: int, arm: str) -> dict:
    import torch
    from transformers import set_seed
    import run_v133_goal_o_budget as budget
    cfg = verify()
    if read(STATE)["status"] != "training" or seed not in (42, 43) or arm not in ("A", "B"):
        raise RuntimeError("AD training gate is not active")
    folder = OUT / f"seed{seed}-{arm}"
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir()
    repair = {x["case_id"]: x for x in rows(OUT / "repair-pairs.jsonl")}
    bindings = {x["case_id"]: repair[x["repair_case_id"]] for x in read(OUT / "repair-update-binding.json")[str(seed)]}
    inputs = {x["id"]: x for x in z.rows(z.OUT / "training-inputs.jsonl")}
    set_seed(seed)
    client, _, _ = z.make_client(SOURCE, z.OUT / "corpus-train.json")
    params = [x[1] for x in client.prepare_for_training()]
    before_base = budget.parameter_hashes(client._model, lora=False)
    before_lora = budget.parameter_hashes(client._model, lora=True)
    op = cfg["optimization"]
    optimizer = torch.optim.AdamW(params, lr=op["learning_rate"], betas=tuple(op["betas"]),
        eps=op["eps"], weight_decay=op["weight_decay"])
    try:
        for step, case_id in enumerate(cfg["order"][str(seed)], 1):
            if step > 128: raise RuntimeError("AD update ceiling exceeded")
            actions = inputs[case_id]["actions"]
            n_tokens = sum(len(x["action_token_ids"]) for x in actions)
            optimizer.zero_grad(set_to_none=True); client.set_training_forward_mode()
            correction = arm == "B" and case_id in bindings
            weight = .5 if correction else 1.
            total_loss = 0.; grad_items = []
            for action in actions:
                lp = client.score_action_tensor([], action["action_token_ids"], True,
                    action["prompt_token_ids"], action["prompt_attention_mask"]).float()
                loss = -weight * lp.sum() / n_tokens
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite original trajectory loss")
                loss.backward(); total_loss += float(loss.detach())
            repair_tokens = 0
            if correction:
                action_items = bindings[case_id]["search_actions"]
                lp_sum = None
                for corrected in action_items:
                    lp = client.score_action_tensor([], corrected["action_token_ids"], True,
                        corrected["prompt_token_ids"], corrected["prompt_attention_mask"]).float()
                    lp_sum = lp.sum() if lp_sum is None else lp_sum + lp.sum()
                    repair_tokens += len(corrected["action_token_ids"])
                loss = -.5 * lp_sum / repair_tokens
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite corrected query loss")
                loss.backward(); total_loss += float(loss.detach())
            grad = float(torch.nn.utils.clip_grad_norm_(params, op["gradient_clip"]))
            if not math.isfinite(grad) or grad <= 0: raise RuntimeError("Invalid gradient")
            budget.guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
            append(folder / "updates.jsonl", {"step": step, "case_id": case_id,
                "loss": total_loss, "gradient": grad, "original_tokens": n_tokens,
                "repair_tokens": repair_tokens, "correction_blended": correction,
                "gpu_memory": client.gpu_memory_stats()})
            if step % 8 == 0: print(json.dumps({"seed": seed, "arm": arm, "updates": step}), flush=True)
        if len(rows(folder / "updates.jsonl")) != 128: raise RuntimeError("Expected exactly 128 updates")
        after_base = budget.parameter_hashes(client._model, lora=False)
        after_lora = budget.parameter_hashes(client._model, lora=True)
        if before_base != after_base: raise RuntimeError("Base weights changed")
        if not any(before_lora[k] != after_lora[k] for k in before_lora): raise RuntimeError("LoRA weights unchanged")
        adapter = folder / "adapter-final"; client._model.save_pretrained(adapter)
        del client, optimizer; gc.collect(); torch.cuda.empty_cache()
        reloaded, _, _ = z.make_client(adapter, z.OUT / "corpus-train.json")
        reloaded._load()
        probe = next((v for v in bindings.values()), None)
        if probe is None: probe = actions[-1]
        action = probe["search_actions"][0] if isinstance(probe, dict) and "search_actions" in probe else probe
        actual = reloaded.score_action_tensor([], action["action_token_ids"], False,
            action["prompt_token_ids"], action["prompt_attention_mask"]).detach().float().cpu()
        del reloaded; gc.collect(); torch.cuda.empty_cache()
        result = {"seed": seed, "arm": arm, "updates": 128, "base_unchanged": True,
            "lora_changed": True, "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
            "reload_checked": True, "probe_logprob_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest()}
        write(folder / "summary.json", result); return result
    except BaseException as exc:
        write(folder / "failure.json", {"error": str(exc), "completed_updates":
            len(rows(folder / "updates.jsonl")) if (folder / "updates.jsonl").exists() else 0})
        raise


def copy_baseline_outputs() -> None:
    for split in SPLITS:
        source = ab.OUT / f"eval-{split}-baseline.jsonl"
        destination = OUT / source.name
        if destination.exists(): raise FileExistsError(destination)
        destination.write_bytes(source.read_bytes())


def evaluate(label: str, adapter: Path, split: str, cases: list[dict], corpus: Path) -> None:
    path = OUT / f"eval-{split}-{label}.jsonl"
    if path.exists(): raise FileExistsError(path)
    tokenizer = load_tokenizer()
    client, wrapped, registry = z.make_client(adapter, corpus)
    for case in cases:
        value = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        gold = case.get("answer", case.get("official_answer"))
        value.update(case_id=case.get("id", case.get("case_id")), split=split, model=label,
            official_answer=gold, **q.score(value["answer"], gold))
        observed_ids = {str(doc.get("id")) for step in value["trace"]
            for doc in step.get("observation", {}).get("result", []) if doc.get("id") is not None}
        observed_titles = {str(doc.get("title")) for step in value["trace"]
            for doc in step.get("observation", {}).get("result", []) if doc.get("title")}
        required_titles = set(map(str, case.get("support_titles", [])))
        required_ids = set(map(str, case.get("source_ids", []))) if not required_titles else set()
        if required_ids or required_titles:
            value["support_retrieved"] = required_ids.issubset(observed_ids) and required_titles.issubset(observed_titles)
            value["required_support_ids"] = sorted(required_ids)
            value["required_support_titles"] = sorted(required_titles)
            answer_prompt = decode_prompt(value["trace"][-1]["prompt_token_ids"], tokenizer)
            answer_docs = visible_documents(answer_prompt)
            answer_ids = {x["id"] for x in answer_docs}
            answer_titles = {x["title"] for x in answer_docs}
            value["support_retrieved_at_answer"] = required_ids.issubset(answer_ids) and required_titles.issubset(answer_titles)
        else:
            value["support_retrieved"] = None
            value["support_retrieved_at_answer"] = None
        append(path, value)
        if len(rows(path)) % 8 == 0: print(json.dumps({"eval": split, "model": label, "completed": len(rows(path))}), flush=True)
    del client, wrapped, registry; gc.collect()
    import torch; torch.cuda.empty_cache()


def evaluate_development() -> dict:
    cfg = verify()
    if read(STATE)["status"] != "training_complete": raise RuntimeError("Train all four models first")
    copy_baseline_outputs()
    for seed in (42, 43):
        for arm in ("A", "B"):
            label = f"{seed}-{'C' if arm == 'A' else 'D'}"
            adapter = OUT / f"seed{seed}-{arm}/adapter-final"
            for split, cases, corpus in ab.evaluation_specs():
                evaluate(label, adapter, split, cases, corpus)
    info = build_evaluation_review_packet("development", ("baseline", "42-C", "43-C", "42-D", "43-D"))
    set_state("awaiting_development_review", development_review=info,
        product_trajectories=4*208)
    return info


def run_training() -> dict:
    if read(STATE)["status"] != "ready_to_train": raise RuntimeError("Material gate did not pass")
    set_state("training")
    results = []
    for seed in (42, 43):
        for arm in ("A", "B"):
            results.append(train(seed, arm))
    set_state("training_complete", training_results=results)
    return evaluate_development()


def candidate_reviewed(stage: str) -> dict[str, dict]:
    packet_path = OUT / f"{stage}-review-packet.jsonl"
    votes_path = OUT / f"{stage}-review-final.jsonl"
    packet, votes = rows(packet_path), rows(votes_path)
    by_id = check_id_rows(packet, votes, "review_id")
    for item in packet:
        vote = by_id[item["review_id"]]
        if vote.get("category") not in aa.CATEGORIES or not vote.get("reason"):
            raise RuntimeError("Invalid semantic evidence review")
        if not set(vote.get("evidence_ids", [])).issubset(item["visible_document_ids"]):
            raise RuntimeError("Review citation references a document absent from the actual prompt")
        if vote.get("category") == "correct_sufficient" and (item["reason"] != "answer" or not vote.get("evidence_ids")):
            raise RuntimeError("A successful answer needs evidence visible at answer time")
        if vote.get("category") == "correct_sufficient":
            doc_map = {d["id"]: d for d in item["steps"][-1]["visible_documents"]}
            citations = vote.get("citations", [])
            if not citations or any(c.get("document_id") not in doc_map or
                not c.get("text") or c["text"] not in doc_map[c["document_id"]]["text"] for c in citations):
                raise RuntimeError("A successful answer needs exact citations from answer-time visible text")
        if not set(vote.get("stop_events", [])).issubset(ac.EVENTS):
            raise RuntimeError("Review contains an unknown search error type")
    return by_id


def build_evaluation_review_packet(stage: str, labels: tuple[str, ...]) -> dict:
    """Freeze anonymous reviews from token-replayed prompts, including clipped passages."""
    tokenizer = load_tokenizer()
    packets, mappings = {}, []
    splits = SPLITS if stage == "development" else ("fresh_confirm",)
    for split in splits:
        for label in labels:
            path = OUT / f"eval-{split}-{label}.jsonl"
            for row in rows(path):
                trace_steps = []
                for step in row.get("trace", []):
                    actual = decode_prompt(step["prompt_token_ids"], tokenizer)
                    trace_steps.append({"action": step.get("action"), "query": step.get("query"),
                        "answer": step.get("answer"), "actual_prompt": actual,
                        "prompt_token_sha256": hashlib.sha256(json.dumps(
                            step["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest(),
                        "visible_documents": visible_documents(actual)})
                if not trace_steps:
                    raise RuntimeError("Evaluation output has no actual model input steps")
                final_prompt = trace_steps[-1]["actual_prompt"]
                visible_ids = sorted({doc["id"] for doc in trace_steps[-1]["visible_documents"]})
                identity = [row["question"], trace_steps, row.get("answer"), row.get("reason")]
                review_id = hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                packets[review_id] = {"review_id": review_id, "question": row["question"],
                    "answer": row.get("answer"), "reason": row.get("reason"),
                    "actual_prompt": final_prompt, "visible_document_ids": visible_ids,
                    "steps": trace_steps,
                    "instructions": "Classify the answer using only actual_prompt and the visible passages in the final step. Inspect earlier steps only to classify search events. Cite exact visible text and document IDs for success. Do not infer from hidden gold, complete untruncated tool logs, model identity, or prior scores."}
                mappings.append({"split": split, "case_id": row["case_id"], "model": label,
                    "review_id": review_id})
    packet_values = sorted(packets.values(), key=lambda x: x["review_id"])
    write_jsonl(OUT / f"{stage}-review-packet.jsonl", packet_values)
    write_jsonl(OUT / f"{stage}-review-map.jsonl", mappings)
    return {"outputs": len(mappings), "unique_reviews": len(packet_values),
        "packet_sha256": sha(OUT / f"{stage}-review-packet.jsonl"),
        "map_sha256": sha(OUT / f"{stage}-review-map.jsonl")}


def candidate_counts(stage: str, labels: tuple[str, ...]) -> tuple[dict, dict]:
    votes = candidate_reviewed(stage)
    mapping = rows(OUT / f"{stage}-review-map.jsonl")
    by_case = {(x["split"], x["model"], x["case_id"]): votes[x["review_id"]] for x in mapping}
    if len(by_case) != len(mapping): raise RuntimeError("Duplicate evaluation review mapping")
    totals = {}
    for split in (("dev", "old", "y", "aa_regression") if stage == "development" else ("fresh_confirm",)):
        for label in labels:
            raw = rows(OUT / f"eval-{split}-{label}.jsonl")
            expected = 64 if split in ("dev", "aa_regression", "fresh_confirm") else 32 if split == "old" else 48
            if len(raw) != expected: raise RuntimeError(f"Wrong output count for {split}/{label}: {len(raw)}")
            c = Counter(by_case[split, label, x["case_id"]]["category"] for x in raw)
            c["invalid"] = sum(x["reason"] == "invalid_action" for x in raw)
            c["em"] = sum(x["em"] for x in raw); c["f1_sum"] = sum(x["f1"] for x in raw)
            c["searches_sum"] = sum(x["searches"] for x in raw)
            c["latency_sum"] = sum(x["elapsed_seconds"] for x in raw)
            c["bad_total"] = c["factual_error"] + c["unsupported_by_visible_evidence"]
            c["wrong_bridge"] = sum("wrong_bridge_query" in by_case[split, label, x["case_id"]].get("stop_events", []) for x in raw)
            c["premature_answer"] = sum("premature_answer" in by_case[split, label, x["case_id"]].get("stop_events", []) for x in raw)
            c["full_evidence_success"] = sum(by_case[split, label, x["case_id"]]["category"] == "correct_sufficient"
                and x.get("support_retrieved_at_answer") is True for x in raw)
            if split == "old":
                groups = {x["case_id"]: x["split"] for x in q.dev_cases()}
                c["main"] = sum(by_case[split, label, x["case_id"]]["category"] == "correct_sufficient"
                    for x in raw if groups[x["case_id"]] == "new_eval")
                c["regression"] = sum(by_case[split, label, x["case_id"]]["category"] == "correct_sufficient"
                    for x in raw if groups[x["case_id"]] == "goal_G_regression")
            totals[f"{split}-{label}"] = dict(c)
    return totals, by_case


def protected_case_ids() -> set[str]:
    return protected_cases_from_baseline("development", "baseline", "aa_regression")


def protected_cases_from_baseline(stage: str, label: str, split: str) -> set[str]:
    """Cases where this baseline succeeds and later searches first supplied required evidence."""
    _, by_case = candidate_counts(stage, (label,))
    cases = (rows(z.OUT / "selected-confirm.jsonl") if split == "aa_regression"
        else rows(ab.OUT / "fresh-confirm-cases.jsonl"))
    required = {x["id"]: (set(map(str, x.get("support_titles", []))) or
        set(map(str, x.get("source_ids", [])))) for x in cases}
    raw = rows(OUT / f"eval-{split}-{label}.jsonl")
    protected = set()
    for row in raw:
        case_id = row["case_id"]
        if by_case[split, label, case_id]["category"] != "correct_sufficient" or \
           row.get("support_retrieved_at_answer") is not True:
            continue
        needed = required.get(case_id, set())
        if not needed:
            continue
        first_docs = row.get("trace", [{}])[0].get("observation", {}).get("result", [])
        all_docs = [d for step in row.get("trace", []) for d in step.get("observation", {}).get("result", [])]
        key = "title" if cases and cases[0].get("support_titles") else "id"
        first = {str(d.get(key)) for d in first_docs}
        later = {str(d.get(key)) for d in all_docs}
        if not needed.issubset(first) and needed.issubset(later):
            protected.add(case_id)
    return protected


def development_decision() -> dict:
    labels = ("baseline", "42-C", "43-C", "42-D", "43-D")
    totals, by_case = candidate_counts("development", labels)
    baseline = {f"{split}-baseline": totals[f"{split}-baseline"] for split in
        ("dev", "old", "y", "aa_regression")}
    gates = {"A": [], "B": []}
    protected = protected_case_ids()
    for arm, candidate_arm in (("A", "C"), ("B", "D")):
        for seed in (42, 43):
            label = f"{seed}-{candidate_arm}"
            dev = totals[f"dev-{label}"]
            old, yv, aar = (totals[f"{sp}-{label}"] for sp in ("old", "y", "aa_regression"))
            lost = sum(by_case["aa_regression", label, case]["category"] != "correct_sufficient" for case in protected)
            base = baseline
            passed = (dev["correct_sufficient"] >= base["dev-baseline"]["correct_sufficient"] + 4 and
                dev["bad_total"] <= base["dev-baseline"]["bad_total"] and dev["invalid"] <= base["dev-baseline"]["invalid"] and
                old["main"] >= base["old-baseline"]["main"]-1 and old["regression"] >= base["old-baseline"]["regression"]-1 and
                yv["correct_sufficient"] >= base["y-baseline"]["correct_sufficient"]-1 and
                aar["correct_sufficient"] >= base["aa_regression-baseline"]["correct_sufficient"]-1 and lost <= 1 and
                read(OUT / f"seed{seed}-{arm}/summary.json")["reload_checked"])
            gates[arm].append({"seed": seed, "passed": passed, "dev_success": dev["correct_sufficient"],
                "dev_bad": dev["bad_total"], "invalid": dev["invalid"], "old_main": old["main"],
                "old_regression": old["regression"], "goal_y": yv["correct_sufficient"],
                "aa_regression": aar["correct_sufficient"], "protected_multihop_lost": lost,
                "wrong_bridge": dev["wrong_bridge"], "full_evidence_success": dev["full_evidence_success"]})
    eligible = {arm for arm, runs in gates.items() if all(x["passed"] for x in runs)}
    mechanism = all(gates["B"][i]["wrong_bridge"] < gates["A"][i]["wrong_bridge"] and
        gates["B"][i]["full_evidence_success"] >= gates["A"][i]["full_evidence_success"] and
        gates["B"][i]["dev_success"] >= gates["A"][i]["dev_success"] for i in (0, 1))
    selected = None
    if eligible:
        selected = sorted(eligible, key=lambda arm: (
            -min(x["dev_success"] for x in gates[arm]),
            -sum(x["dev_success"] for x in gates[arm])/2,
            sum(x["dev_bad"] for x in gates[arm]),
            0 if arm == "A" else 1))[0]
    decision = {"status": "confirmation_authorized" if selected else "complete_development_gate_failed",
        "gates": gates, "selected_arm": selected, "mechanism_supported": mechanism,
        "counts": totals, "protected_case_ids": sorted(protected)}
    write(OUT / "development-decision.json", decision)
    set_state(decision["status"], development_decision=decision)
    if decision["status"] == "complete_development_gate_failed":
        write_readme_report({"published": False, "product_adapter": str(SOURCE), "gates": gates})
    return decision


def evaluate_confirmation() -> dict:
    cfg = verify()
    if read(STATE)["status"] != "confirmation_authorized": raise RuntimeError("Confirmation gate is closed")
    manifest = read(ab.OUT / "fresh-confirm-manifest.json")
    if manifest.get("confirmation_accessed"): raise RuntimeError("Fresh confirmation already accessed")
    cases, corpus = rows(ab.OUT / "fresh-confirm-cases.jsonl"), ab.OUT / "fresh-confirm-corpus.json"
    if len(cases) != 64 or sha(corpus) != cfg["confirmation_sha256"]["corpus"]:
        raise RuntimeError("Frozen confirmation changed")
    outputs = []
    selected = read(OUT / "development-decision.json")["selected_arm"]
    for label, adapter in [("baseline", SOURCE), ("42", OUT / f"seed42-{selected}/adapter-final"),
                           ("43", OUT / f"seed43-{selected}/adapter-final")]:
        evaluate(f"confirm-{label}", adapter, "fresh_confirm", cases, corpus)
        outputs.extend(rows(OUT / f"eval-fresh_confirm-confirm-{label}.jsonl"))
    if len(outputs) != 192: raise RuntimeError("Confirmation output ceiling or count mismatch")
    info = build_evaluation_review_packet("confirmation", ("confirm-baseline", "confirm-42", "confirm-43"))
    manifest["confirmation_accessed"] = True
    write(ab.OUT / "fresh-confirm-manifest.json", manifest)
    info.update(outputs=192, selected_arm=selected)
    set_state("awaiting_confirmation_review", confirmation_review=info, confirmation_accessed=True)
    return info


def confirmation_decision() -> dict:
    labels = ("confirm-baseline", "confirm-42", "confirm-43")
    totals, by_case = candidate_counts("confirmation", labels)
    base = totals["fresh_confirm-confirm-baseline"]
    selected = read(OUT / "development-decision.json")["selected_arm"]
    gates = []
    protected = protected_cases_from_baseline("confirmation", "confirm-baseline", "fresh_confirm")
    for seed in (42, 43):
        label = f"confirm-{seed}"
        c = totals[f"fresh_confirm-{label}"]
        lost = sum(by_case["fresh_confirm", label, case]["category"] != "correct_sufficient" for case in protected
            if case in by_case["fresh_confirm", label])
        gates.append({"seed": seed, "passed": c["correct_sufficient"] >= base["correct_sufficient"]+4 and
            c["bad_total"] <= base["bad_total"] and c["invalid"] <= base["invalid"] and lost <= 1,
            "success": c["correct_sufficient"], "baseline_success": base["correct_sufficient"],
            "bad_total": c["bad_total"], "baseline_bad_total": base["bad_total"],
            "invalid": c["invalid"], "protected_lost": lost})
    passed = all(x["passed"] for x in gates)
    route = "A" if selected == "A" else "B"
    adapter = str(OUT / f"seed42-{route}/adapter-final") if passed else str(SOURCE)
    result = {"status": "complete_confirmed_and_published" if passed else "complete_confirmation_gate_failed",
        "published": passed, "selected_arm": selected if passed else None,
        "product_adapter": adapter, "gates": gates, "counts": totals}
    write(OUT / "product-decision.json", result); set_state(result["status"], product_decision=result)
    if passed:
        q_decision_path = q.OUT / "decision.json"
        if not q_decision_path.is_file():
            raise RuntimeError("Cannot publish Goal AD: Goal Q product selection record is absent")
        old = read(q_decision_path)
        write(OUT / "prior-goal-q-product-decision.json", old)
        write(q_decision_path, {"selected": "goal_ad", "adapter": adapter,
            "confirmed": True, "reason": "goal_ad_confirmation_gate_passed",
            "prior_selection": old,
            "goal_ad_confirmation_sha256": sha(OUT / "confirmation-review-final.jsonl")})
    write_readme_report(result)
    return result


def write_readme_report(decision: dict | None = None) -> None:
    status = read(STATE)["status"]
    if decision is None:
        decision = read(OUT / "product-decision.json") if (OUT / "product-decision.json").exists() else {}
    report = ["# Goal AD — Bridge query training", "", f"Status: `{status}`", "",
        f"Published: `{decision.get('published', False)}`", f"Selected adapter: `{decision.get('product_adapter', SOURCE)}`", "",
        "The experiment uses the Goal Z trajectory seed 42 checkpoint and the frozen Goal Q search path.",
        "It only adds search-action supervision at audited failure states; it does not supervise the corrected answer.", ""]
    if (OUT / "diagnostic-gate.json").exists(): report += ["## Diagnostic gate", "", "```json", json.dumps(read(OUT / "diagnostic-gate.json"), ensure_ascii=False, indent=2), "```", ""]
    if (OUT / "repair-material-decision.json").exists(): report += ["## Repair material gate", "", "```json", json.dumps(read(OUT / "repair-material-decision.json"), ensure_ascii=False, indent=2), "```", ""]
    if (OUT / "development-decision.json").exists(): report += ["## Development decision", "", "```json", json.dumps(read(OUT / "development-decision.json"), ensure_ascii=False, indent=2), "```", ""]
    if decision.get("gates"): report += ["## Product decision", "", "```json", json.dumps(decision["gates"], ensure_ascii=False, indent=2), "```", ""]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def summarize() -> dict:
    if not STATE.exists():
        return {"status": "not_prepared"}
    state = read(STATE)
    summary = {"state": state, "confirmation_accessed": state.get("confirmation_accessed", False)}
    if (OUT / "diagnostic-gate.json").exists(): summary["diagnostic_gate"] = read(OUT / "diagnostic-gate.json")
    if (OUT / "query-proposal-gate.json").exists(): summary["query_gate"] = read(OUT / "query-proposal-gate.json")
    if (OUT / "repair-material-decision.json").exists(): summary["repair_gate"] = read(OUT / "repair-material-decision.json")
    if (OUT / "development-decision.json").exists(): summary["development_decision"] = read(OUT / "development-decision.json")
    if (OUT / "product-decision.json").exists(): summary["product_decision"] = read(OUT / "product-decision.json")
    write_readme_report(summary.get("product_decision", {
        "published": False, "product_adapter": str(SOURCE),
        "gates": summary.get("development_decision", {}).get("gates", {})}))
    write(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    value = {"prepare": prepare, "run": run, "summarize": summarize}[args.command]()
    if value is not None and args.command != "summarize":
        print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
