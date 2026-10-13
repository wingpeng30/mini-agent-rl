"""Verify and seal the CPU-only Goal-Z 64-case admission audit."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/v133-goal-ai-z64-source-admission-v2"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lines(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    prep_freeze = json.loads((RUN / "freeze.json").read_text(encoding="utf-8"))
    fixed_freeze = json.loads((ROOT / "runs/v133-goal-ai-fixed-z-coverage-v1/freeze.json").read_text(encoding="utf-8"))
    cases = lines(ROOT / "runs/v133-goal-ah-entity-relation/evaluation-cases-goal_z_dev.jsonl")
    wanted = {x["case_id"]: x for x in cases}
    source = {}
    srcdir = ROOT.parent / "goal-z-sources"
    with zipfile.ZipFile(srcdir / "musique_v1.0.zip") as z:
        for line in z.open("data/musique_ans_v1.0_dev.jsonl"):
            x = json.loads(line)
            key = "musique:" + x["id"]
            if key in wanted:
                source[key] = (x["question"], x["answer"])
    with zipfile.ZipFile(srcdir / "2wiki_data_ids_april7.zip") as z:
        for x in json.load(z.open("dev.json")):
            key = "2wiki:" + x["_id"]
            if key in wanted:
                source[key] = (x["question"], x["answer"])
    if len(cases) != 64 or set(source) != set(wanted):
        raise RuntimeError("Official record count or ID mismatch")
    if any(source[k] != (v["question"], v["answer"]) for k, v in wanted.items()):
        raise RuntimeError("Literal question/answer mismatch")
    idsha = hashlib.sha256("\n".join(x["case_id"] for x in cases).encode()).hexdigest()
    if idsha != prep_freeze["case_ids_order_sha256"]:
        raise RuntimeError("Frozen case order/hash mismatch")

    admission = lines(RUN / "source-admission-ledger.final.jsonl")
    edges = lines(RUN / "edge-evidence-ledger.final.jsonl")
    mechanisms = lines(RUN / "trajectory-mechanism-ledger.jsonl")
    packets = lines(RUN / "official-source-packets.jsonl")
    if len(admission) != 64 or len(edges) != 128 or len(mechanisms) != 64 or len(packets) != 64:
        raise RuntimeError("Ledger cardinality mismatch")
    if [x["case_id"] for x in admission] != [x["case_id"] for x in cases]:
        raise RuntimeError("Admission order mismatch")
    if not all(x.get("diagnostic_only") and not x.get("training_allowed") for x in admission):
        raise RuntimeError("Evaluation-isolation marker missing")
    if not all(x.get("exact_body_preserved") for p in packets for x in p["official_support_documents"]):
        raise RuntimeError("Official support body missing from frozen corpus")
    if any(sum(x[k] for x in mechanisms) for k in ("new_searches", "new_model_calls", "gpu_calls", "updates")):
        raise RuntimeError("A prohibited action was recorded")

    adapter = ROOT / "runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final/adapter_model.safetensors"
    actual_adapter = digest(adapter)
    expected_adapter = fixed_freeze["hashes"]["runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final/adapter_model.safetensors"]
    if actual_adapter != expected_adapter or actual_adapter != "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52":
        raise RuntimeError("Goal Z adapter identity mismatch")

    for row in admission:
        q, a = source[row["case_id"]]
        row["source_record_exact_question"] = q == row["question"]
        row["source_record_exact_answer"] = a == row["official_answer"]
    write_jsonl(RUN / "source-admission-ledger.final.jsonl", admission)
    revisions = RUN / "review-revisions.jsonl"
    if not revisions.exists():
        revisions.write_text("", encoding="utf-8")
    (RUN / "source-identity-verification.json").write_text(json.dumps({
        "case_count": 64, "literal_question_matches": 64, "literal_answer_matches": 64,
        "ids_and_order_sha256": idsha, "source_archives_sha256": prep_freeze["source_archives_sha256"],
        "result": "pass"}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    (RUN / "adjudication-freeze.json").write_text(json.dumps({
        "review_version": "z64-one-pass-v1",
        "prepared_input_freeze_sha256": digest(RUN / "freeze.json"),
        "preparer_sha256": digest(ROOT / "scripts/run_v133_z64_source_admission.py"),
        "finalizer_sha256": digest(ROOT / "scripts/finalize_v133_z64_source_admission.py"),
        "verifier_sha256": digest(Path(__file__)),
        "case_ids_and_order_sha256": idsha,
        "source_archives_sha256": prep_freeze["source_archives_sha256"],
        "frozen_dev_corpus_sha256": prep_freeze["files_sha256"]["mini-agent-rl/runs/v133-goal-z-multihop-sft/corpus-dev.json"],
        "adapter_sha256_expected": expected_adapter, "adapter_sha256_actual": actual_adapter,
        "adapter_loaded": False, "candidate_cap_per_edge": 12,
        "candidate_review_scope": "For new cases, one official support body was reviewed per edge. Remaining corpus candidates were not scanned; unconfirmed is bounded and does not establish corpus-wide absence.",
        "new_searches": 0, "new_model_calls": 0, "gpu_calls": 0, "training_updates": 0,
        "diagnostic_only": True, "training_allowed": False
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify = {"status": "pass", "case_count": 64, "ordered_ids_match": True,
              "literal_question_answer_matches": 64, "official_records": 64,
              "support_bodies": 128, "support_bodies_exact_in_corpus": 128,
              "edge_review_rows": 128, "mechanism_rows": 64,
              "admitted_cases": sum(x["admission"] == "admitted" for x in admission),
              "adapter_expected_sha256": expected_adapter, "adapter_actual_sha256": actual_adapter,
              "adapter_loaded": False, "new_searches": 0, "model_calls": 0,
              "gpu_calls": 0, "updates": 0, "training_allowed": False}
    (RUN / "verification.json").write_text(json.dumps(verify, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    file_hashes = {p.name: digest(p) for p in sorted(RUN.iterdir())
                   if p.is_file() and p.name != "delivery-manifest.json"}
    (RUN / "delivery-manifest.json").write_text(json.dumps({
        "sha256": file_hashes, "final_audit_code_sha256": digest(ROOT / "scripts/finalize_v133_z64_source_admission.py"),
        "verifier_sha256": digest(Path(__file__))}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verify, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
