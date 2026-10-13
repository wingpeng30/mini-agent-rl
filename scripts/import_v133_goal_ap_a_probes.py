"""Safely reuse only valid legacy A probe generations from invalid AP v1."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "runs/v133-goal-ap-evidence-presentation-v1"
V2 = ROOT / "runs/v133-goal-ap-evidence-presentation-v2"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_once(path: Path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in value),
                    encoding="utf-8", newline="\n")


def main():
    m1 = read_json(V1 / "manifest.json")
    m2 = read_json(V2 / "manifest.json")
    if m1["expected_hashes"] != m2["expected_hashes"]:
        raise RuntimeError("adapter identities differ between AP runs")
    frozen = read_jsonl(V2 / "probe-inputs.jsonl")
    expected = {(r["model"], r["probe_id"]): r for r in frozen if r["arm"] == "A"}
    source_hashes = {}
    output_counts = {}
    for model in ("AM", "Z"):
        src = V1 / f"probe-{model}-A.jsonl"
        dst = V2 / f"probe-{model}-A.jsonl"
        rows = read_jsonl(src)
        if len(rows) != 32 or len({r["probe_id"] for r in rows}) != 32:
            raise RuntimeError(f"v1 legacy A output set incomplete for {model}")
        for row in rows:
            inp = expected[(model, row["probe_id"])]
            if row.get("arm") != "A" or row.get("prompt_token_ids") != inp["prompt_token_ids"]:
                raise RuntimeError(f"legacy A input mismatch: {model}/{row['probe_id']}")
            if row.get("no_search_execution") is not True:
                raise RuntimeError("source A probe unexpectedly executed a search")
        write_once(dst, rows)
        source_hashes[model] = hashlib.sha256(src.read_bytes()).hexdigest()
        output_counts[model] = len(rows)
    provenance = {
        "status": "reused_valid_legacy_A_only",
        "v1_run_manifest_sha256": hashlib.sha256((V1 / "manifest.json").read_bytes()).hexdigest(),
        "v2_run_manifest_sha256": hashlib.sha256((V2 / "manifest.json").read_bytes()).hexdigest(),
        "v1_a_output_sha256": source_hashes,
        "reused_outputs": output_counts,
        "excluded": ["probe-AM-B.jsonl", "probe-Z-B.jsonl"],
        "validation": [
            "same adapter hashes in both manifests",
            "every A prompt token id matches the corrected v2 A input",
            "no search was executed during reused probes",
        ],
    }
    p = V2 / "reused-a-provenance.json"
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8", newline="\n")
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
