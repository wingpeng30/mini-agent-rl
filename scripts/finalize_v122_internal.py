"""Fail-closed merge and freeze of the new v122 internal evaluation set."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from mini_agent_rl.tools.search import _BM25Index
from scripts.build_v114_sft_candidates import norm
from scripts.finalize_v116 import export
from scripts.finalize_v117 import RISKS, valid_v117_vote
from scripts.prepare_v116_data import ROOT, full_rows, sha, write, writel

OUT = ROOT / "data/evidence-sft-v122-internal-candidates"
REVIEW = OUT / "review"
FROZEN = OUT / "frozen"


def loadl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def choose(ordered: list[tuple[str, list]], count: int = 50):
    if len(ordered) < count:
        return None
    risks = [Counter(tag for _, labels in records for tag in labels) for _, records in ordered]
    suffix = [Counter() for _ in range(len(ordered) + 1)]
    for i in range(len(ordered) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + risks[i]
    selected: list[tuple[str, list]] = []
    failed = set()

    def visit(i: int, need: tuple[int, ...]) -> bool:
        if len(selected) == count:
            return not any(need)
        if len(ordered) - i < count - len(selected) or any(suffix[i][r] < need[j] for j, r in enumerate(RISKS)):
            return False
        key = (i, count - len(selected), need)
        if key in failed:
            return False
        selected.append(ordered[i])
        next_need = tuple(max(0, need[j] - risks[i][r]) for j, r in enumerate(RISKS))
        if visit(i + 1, next_need):
            return True
        selected.pop()
        if visit(i + 1, need):
            return True
        failed.add(key)
        return False

    return list(selected) if visit(0, (5, 5, 5, 5)) else None


def main() -> None:
    history = json.loads((OUT / "history-inventory.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "candidate-freeze.json").read_text(encoding="utf-8"))
    manifest = json.loads((REVIEW / "manifest.json").read_text(encoding="utf-8"))
    tokens = json.loads((OUT / "token-replay.json").read_text(encoding="utf-8"))
    if not history.get("ready") or history.get("failures") or history.get("sealed_content_read"):
        raise RuntimeError("historical_exposure_gate_failed")
    if sha(OUT / "candidate-corpus-private.json") != freeze["corpus_sha256"] or freeze["corpus_sha256"] != manifest["corpus_sha256"]:
        raise RuntimeError("candidate_corpus_changed")
    if sha(REVIEW / "internal-packets.jsonl") != manifest["packet_sha256"] or not tokens.get("all_exact") or tokens.get("packet_sha256") != manifest["packet_sha256"]:
        raise RuntimeError("final_input_or_token_replay_changed")
    packets = loadl(REVIEW / "internal-packets.jsonl")
    if len(packets) != manifest["states"] or len({p["case_id"] for p in packets}) != len(packets):
        raise RuntimeError("packet_coverage_or_duplicate")
    source_map = {case: row["source_id"] for row in loadl(REVIEW / "eligible-private.jsonl") for case in row["case_ids"]}
    if set(source_map) != {p["case_id"] for p in packets}:
        raise RuntimeError("private_source_mapping_invalid")
    api: dict[str, dict] = {}
    luna: dict[str, dict] = {}
    luna_batches: list[str] = []
    luna_gap_seen = False
    for batch, spec in manifest["batches"].items():
        path = REVIEW / batch
        batch_packets = loadl(path)
        expected_cases = {p["case_id"] for p in batch_packets}
        if sha(path) != spec["sha256"] or len(batch_packets) != spec["states"] or spec["states"] > 20:
            raise RuntimeError("review_batch_changed")
        for prefix, result in (("api", api), ("luna", luna)):
            review_path = REVIEW / f"{prefix}-{batch}"
            if not review_path.exists():
                if prefix == "luna":
                    luna_gap_seen = True
                continue
            if prefix == "luna":
                if luna_gap_seen:
                    raise RuntimeError("non_contiguous_luna_review_batches")
                if len(loadl(review_path)) != spec["states"]:
                    raise RuntimeError("partial_luna_review_batch")
                luna_batches.append(batch)
            reviewed_rows = loadl(review_path)
            if {row.get("case_id") for row in reviewed_rows} != expected_cases:
                raise RuntimeError(f"{prefix}_batch_case_coverage_mismatch")
            for row in reviewed_rows:
                case = row.get("case_id")
                if not case or case in result:
                    raise RuntimeError(f"duplicate_{prefix}_review")
                if prefix == "api" and row.get("input_file_sha256") != spec["sha256"]:
                    raise RuntimeError("api_batch_hash_mismatch")
                result[case] = row
    approved: dict[str, list] = defaultdict(list)
    details: list[dict] = []
    for packet in packets:
        case = packet["case_id"]
        lrow = luna.get(case)
        arow = api.get(case)
        lv, lr = valid_v117_vote(packet, lrow)
        av, ar = valid_v117_vote(packet, arow.get("decision") if arow else None)
        if lrow and lrow.get("packet_sha256") != packet["packet_sha256"]:
            lv, lr = False, "luna_packet_hash_mismatch"
        if arow and arow.get("packet_sha256") != digest(packet):
            av, ar = False, "api_packet_hash_mismatch"
        accepted = bool(lv and av and lrow["approved"] and arow["decision"]["approved"])
        labels = sorted(set(lrow["risk_labels"]) & set(arow["decision"]["risk_labels"])) if accepted else []
        if accepted:
            approved[source_map[case]].append((packet, labels))
        details.append({"case_id": case, "source_id": source_map[case], "accepted": accepted,
                        "luna_validation": lr, "api_validation": ar,
                        "luna_approved": lrow.get("approved") if lv else None,
                        "api_approved": arow["decision"].get("approved") if av else None,
                        "risk_labels": labels})
    pairs = [(qid, approved[qid]) for qid in freeze["ids"] if len(approved[qid]) == 2 and
             {p["proposed_action"]["action"] for p, _ in approved[qid]} == {"search", "answer"}]
    selected = choose(pairs)
    risk_counts = Counter(tag for _, records in selected or [] for _, labels in records for tag in labels)
    prior = json.loads((OUT / "history-private.json").read_text(encoding="utf-8"))
    selected_ids = {qid for qid, _ in selected or []}
    selected_text = {norm(p["question"]) for _, records in selected or [] for p, _ in records}
    isolated = not (selected_ids & set(prior["ids"]) or selected_text & set(prior["normalized_questions"]))
    index = _BM25Index(json.loads((OUT / "candidate-corpus-private.json").read_text(encoding="utf-8")))
    def project(docs):
        return [{k: doc[k] for k in ("id", "title", "text")} for doc in docs]
    retrieval_exact = bool(selected) and all(project(index.search(p["proposed_action"]["query"], top_k=3)) == p["retrieved_after_search"]
                                           for _, records in selected for p, _ in records if p["proposed_action"]["action"] == "search")
    reviewed_prefix = {p["case_id"] for batch in luna_batches for p in loadl(REVIEW / batch)}
    checks = {"contiguous_complete_double_reviewed_prefix": bool(luna_batches) and
              set(luna) == reviewed_prefix and reviewed_prefix <= set(api),
              "50_complete_pairs": bool(selected) and len(selected) == 50,
              "four_risks_five_states_each": all(risk_counts[r] >= 5 for r in RISKS),
              "historical_question_isolation": isolated,
              "top3_replay_exact": retrieval_exact,
              "unique_final_states": bool(selected) and len({digest(export(p)) for _, records in selected for p, _ in records}) == 100}
    report = {"version": "v122-new-internal-gate-v1", "checks": checks,
              "local_internal_ready": all(checks.values()), "training_allowed": False,
              "candidate_questions": freeze["source_count"], "candidate_states": len(packets),
              "api_reviewed": len(api), "luna_reviewed": len(luna),
              "luna_batches": luna_batches,
              "approved_pairs": len(pairs), "selected_pairs": len(selected or []),
              "selected_risks": dict(risk_counts),
              "gaps": {"pairs": max(0, 50 - len(pairs)),
                       "risks": {r: max(0, 5 - risk_counts[r]) for r in RISKS},
                       "unreviewed": len(packets) - min(len(api), len(luna))},
              "independent100_accessed": False, "final_test_accessed": False}
    write(ROOT / "reports/v122-internal-preflight.json", report)
    writel(REVIEW / "merge-details.jsonl", details)
    if not report["local_internal_ready"]:
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return
    FROZEN.mkdir(parents=True, exist_ok=True)
    writel(FROZEN / "internal.jsonl", [export(p) for _, records in selected for p, _ in records])
    sources = full_rows([qid for qid, _ in selected])
    tasks = [{"id": qid, "question": sources[qid]["question"], "answer": sources[qid]["answer"],
              "metadata": {"source": "hotpotqa-official-train", "source_id": qid}}
             for qid, _ in selected]
    states = [{"state_id": p["case_id"], "task_id": qid, "question": p["question"],
               "messages": p["visible_messages"], "target": p["proposed_action"],
               "evidence": p["before_action"], "retrieval_top3": p.get("retrieved_after_search") or []}
              for qid, records in selected for p, _ in records]
    writel(FROZEN / "internal-tasks-private.jsonl", tasks)
    writel(FROZEN / "internal-states-private.jsonl", states)
    (FROZEN / "internal-corpus.json").write_bytes((OUT / "candidate-corpus-private.json").read_bytes())
    write(FROZEN / "manifest.json", {"version": "v122-new-internal-freeze-v1",
          "questions": 50, "states": 100, "corpus_documents": freeze["documents"],
          "selected_ids_sha256": digest([qid for qid, _ in selected]),
          "files": {path.name: sha(path) for path in FROZEN.iterdir() if path.is_file()},
          "training_allowed": False, "remote_preflight_pending": True})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
