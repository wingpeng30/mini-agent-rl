"""Freeze blinded quality-audit packets for deterministic internal question replacements."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from build_v114_sft_candidates import ROOT, OUT as CLEAN, TRAIN_FILES, collect_history, norm, risk_tags
from prepare_v114_reviews import canonical, sha

WORK = ROOT / "data/evidence-sft-v114-review"
SEED = "v114-internal-replacement-proposals-v3:"
MEASURE_REQUEST = re.compile(
    r"\bhow many\b|\bhow much\b|\bhow long\b|\bwhat number\b|\bpopulation\b|\binhabitants\b|"
    r"\blength\b|\bheight\b|\bweight\b|\bdistance\b|\bpercentage\b|\bpercent\b|\barea of\b|\bduration\b",
    re.I,
)
SUBJECTIVE = re.compile(r"\b(more well[- ]known|better known|most famous|more famous|best known)\b", re.I)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError(f"immutable_artifact_changed:{path.name}")
    if not path.exists():
        path.write_bytes(raw)


def main() -> None:
    manifest = load_json(CLEAN / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha((CLEAN / name).read_bytes()) != expected:
            raise RuntimeError(f"source_manifest_hash_mismatch:{name}")
    old_internal = set(load_json(CLEAN / "internal-freeze.json")["ids"])
    v2_freeze = load_json(WORK / "internal-freeze-v2.json")
    old_internal |= set(v2_freeze["ids"])
    old_train = set(load_json(CLEAN / "training-question-freeze.json")["ids"])
    reserve = load_jsonl(WORK / "reserve-private.jsonl")
    old_train |= {row["source_id"] for row in reserve if row["split"] == "train"}
    old_internal |= {row["source_id"] for row in reserve if row["split"] == "internal"}
    history, _ = collect_history()
    history_norms = {row for row in history if not row.startswith("__id__")}
    history_ids = {row.removeprefix("__id__") for row in history if row.startswith("__id__")}

    unique: dict[str, dict] = {}
    for path in TRAIN_FILES:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["id", "question"], batch_size=4096):
            for row in batch.to_pylist():
                qid, question = str(row["id"]), row["question"]
                qnorm = norm(question)
                if (not isinstance(question, str) or len(question) > 250 or not MEASURE_REQUEST.search(question)
                        or SUBJECTIVE.search(question) or qid in old_internal | old_train
                        or qid.casefold() in history_ids or qnorm in history_norms or qnorm in unique):
                    continue
                if "measurement" not in risk_tags(question):
                    continue
                unique[qnorm] = {"source_id": qid, "question": question}
    candidates = sorted(unique.values(), key=lambda row: (
        hashlib.sha256((SEED + row["source_id"]).encode()).hexdigest(), row["source_id"]))
    selected = candidates[:12]
    if len(selected) != 12:
        raise RuntimeError(f"replacement_proposal_pool_too_small:{len(selected)}")
    wanted = {row["source_id"] for row in selected}
    contexts: dict[str, dict] = {}
    for path in TRAIN_FILES:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["id", "question", "context"], batch_size=512):
            for row in batch.to_pylist():
                qid = str(row["id"])
                if qid in wanted:
                    if row["question"] != next(item["question"] for item in selected if item["source_id"] == qid):
                        raise RuntimeError("replacement_question_changed_between_passes")
                    context = row["context"]
                    docs = [{"title": title, "text": " ".join(sentences)}
                            for title, sentences in zip(context["title"], context["sentences"])]
                    contexts[qid] = {"question": row["question"], "context": docs}
    if set(contexts) != wanted:
        raise RuntimeError("replacement_public_context_missing")

    public, private = [], []
    for ordinal, meta in enumerate(selected):
        source_id = meta["source_id"]
        context = contexts[source_id]
        audit_id = hashlib.sha256(("v114-replacement-question-audit:" + source_id).encode()).hexdigest()[:20]
        row = {"audit_id": audit_id, "question": context["question"], "context": context["context"]}
        row["packet_sha256"] = sha(canonical(row))
        public.append(row)
        private.append({"audit_id": audit_id, "order": ordinal, "source_id": source_id,
                        "normalized_question_sha256": sha(norm(context["question"]).encode())})
    payload = b"".join(canonical(row) + b"\n" for row in public)
    private_payload = b"".join(canonical(row) + b"\n" for row in private)
    for reviewer in ("a", "b"):
        write_once(WORK / f"internal-replacement-proposals-{reviewer}-packets.jsonl", payload)
    write_once(WORK / "internal-replacement-proposals-private.jsonl", private_payload)
    summary = {"version": "v114-internal-replacement-proposals-v1", "count": len(public),
               "selection_seed": SEED, "selection_rule": "first 12 unique, historically unseen official-training questions requesting a numeric measurement, ordered by SHA256(seed + source_id); no answer/model/evaluation field read",
               "source_manifest_sha256": sha((CLEAN / "manifest.json").read_bytes()),
               "private_mapping_sha256": sha(private_payload), "public_packet_sha256": sha(payload),
               "independent100_content_read": False, "final_test_accessed": False}
    write_once(WORK / "internal-replacement-proposals-manifest.json",
               (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"proposals": len(public), "packet_sha256": sha(payload)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
