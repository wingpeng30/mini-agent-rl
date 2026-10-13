"""Recompute primary-review coverage, citation binding, and effective amendments."""
from collections import Counter
import json
from scripts.v123_cpu_core import digest, readl, write
from scripts.v124_forensic import OUT
from scripts.v124_primary_batch001 import evidence_binding


def verify():
    inventory = readl(OUT / "inventory-private.jsonl")
    by_id = {r["record_id"]: r for r in inventory}
    assert len(by_id) == len(inventory)
    lineage = {r["record_id"]: r for r in readl(OUT / "lineage-private.jsonl")}
    decisions = [r for path in sorted(OUT.glob("primary-batch-*.jsonl")) for r in readl(path)]
    by_review = {r["record_id"]: r for r in decisions}
    assert len(by_review) == len(decisions)
    for review in decisions:
        source = by_id[review["record_id"]]
        assert review["record_sha256"] == source["record_sha256"]
        assert not review["independent_review"] and not review["approved_training_label"]
        assert not review["training_allowed"]
        bound, retrieved = evidence_binding(source, lineage)
        assert review["lineage_binding_sha256"] == digest(bound)
        for cite in review["citations"]:
            assert cite["stage"] in {"before_action", "retrieved_after_search"}
            documents = (source["before_action"] if cite["stage"] == "before_action"
                         else retrieved)
            assert any(d["id"] == cite["document_id"] and cite["quote"] in d["text"]
                       for d in documents), review["record_id"]
    amendments = [r for path in sorted(OUT.glob("primary-amendments*-private.jsonl"))
                  for r in readl(path)]
    assert len({r["record_id"] for r in amendments}) == len(amendments)
    effective = dict(by_review)
    for change in amendments:
        original = by_review[change["record_id"]]
        assert change["record_sha256"] == original["record_sha256"]
        assert change["superseded_primary_review_sha256"] == digest(original)
        assert not change["independent_review"] and not change["approved_training_label"]
        assert not change["training_allowed"]
        effective[change["record_id"]] = {**original, "disposition": change["effective_disposition"]}
    adjudications_path = OUT / "primary-adjudication-overrides-private.jsonl"
    adjudications = readl(adjudications_path) if adjudications_path.exists() else []
    if len({row["record_id"] for row in adjudications}) != len(adjudications):
        raise ValueError("duplicate_primary_adjudication_override")
    for row in adjudications:
        record_id = row["record_id"]
        if record_id not in effective or row["record_sha256"] != by_id[record_id]["record_sha256"]:
            raise ValueError("primary_adjudication_source_mismatch:" + record_id)
        if row["supersedes_effective_disposition"] != effective[record_id]["disposition"]:
            raise ValueError("primary_adjudication_chain_mismatch:" + record_id)
        if (row.get("independent_review") is not True or row.get("approved_training_label") is not False
                or row.get("training_allowed") is not False or not row.get("basis")):
            raise ValueError("invalid_primary_adjudication_metadata:" + record_id)
        effective[record_id] = {**effective[record_id],
                                "disposition": row["effective_disposition"]}
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
    assert progress["primary_reviewed"] == len(decisions)
    assert progress["primary_remaining"] == len(inventory) - len(decisions)
    assert not progress["training_allowed"] and progress["repaired_double_approved"] == 0
    result = {"reviewed": len(decisions), "total": len(inventory),
              "effective_dispositions": dict(Counter(r["disposition"] for r in effective.values())),
              "amendments": len(amendments), "adjudication_overrides": len(adjudications),
              "citation_bindings_checked": sum(len(r["citations"]) for r in decisions),
              "next_record": progress["next_record"], "training_allowed": False}
    write(OUT / "primary-verification.json", result)
    return result


if __name__ == "__main__":
    print(verify())
