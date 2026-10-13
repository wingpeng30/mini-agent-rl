"""Freeze supplementary questions without consulting retrieval or model results."""
from collections import Counter
import json
from pathlib import Path
import random
import re
import subprocess
from uuid import uuid4

from mini_agent_rl.retrieval.local import ROOT, config, corpus, sha
from prepare_v133_goal_z_multihop_data import audit_case, norm


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    source = ROOT / "runs/v133-goal-z-multihop-sft"
    cfg = config()
    documents = corpus(cfg)
    evidence = {(d["title"], d["text"]): d["id"] for d in documents}
    previous = [r for name in ("selected-train.jsonl", "selected-dev.jsonl", "selected-confirm.jsonl") for r in rows(source / name)]
    old_ids = {r["id"] for r in previous}
    old_questions = {norm(r["question"]) for r in previous}
    forbidden = {r["id"] for name in ("train-semantic-rejections.jsonl", "eval-semantic-rejections.jsonl", "source-audit-rejections.jsonl") for r in rows(source / name)}
    pool, rejected = [], []
    seen = set(old_questions)
    for name in ("candidates-dev-musique.jsonl", "candidates-dev-2wiki.jsonl"):
        for case in rows(source / name):
            supports = [d for d in case["docs"] if d["support"]]
            reason = None
            if case["id"] in old_ids:
                reason = "already_selected_split"
            elif case["id"] in forbidden:
                reason = "previous_quality_rejection"
            elif norm(case["question"]) in seen:
                reason = "duplicate_normalized_question"
            elif not supports or any((d["title"], d["text"]) not in evidence for d in supports):
                reason = "exact_support_not_in_current_corpus"
            else:
                reason = audit_case(case, (set(), set(), set()))
            if reason:
                rejected.append(dict(id=case["id"], reason=reason))
                continue
            seen.add(norm(case["question"]))
            pool.append({k: v for k, v in case.items() if k != "docs"} | dict(
                support_titles=[d["title"] for d in supports],
                support_document_ids=[evidence[d["title"], d["text"]] for d in supports],
                support_documents=supports))
    rng = random.Random(20261006)
    selected = []
    for label in ("musique", "2wiki"):
        candidates = sorted((r for r in pool if r["source"] == label), key=lambda r: r["id"])
        if len(candidates) < 32:
            raise ValueError(f"Quality-screened pool shortfall: {label}={len(candidates)}")
        selected.extend(rng.sample(candidates, 32))
    rng.shuffle(selected)
    folder = Path(cfg["run_dir"]) / ("supplement-64-" + uuid4().hex)
    folder.mkdir(parents=True)
    pattern_path = folder / "candidate-ids.txt"
    pattern_path.write_text("\n".join(r["id"] for r in pool) + "\n", encoding="utf-8")
    scan = subprocess.run(["rg", "-l", "-F", "-f", str(pattern_path), "-g", "*.json", "-g", "*.jsonl", str(ROOT / "runs")], capture_output=True, text=True, encoding="utf-8")
    if scan.returncode not in (0, 1):
        raise RuntimeError(scan.stderr)
    regex = re.compile("|".join(re.escape(r["id"]) for r in pool))
    exposure = {r["id"]: [] for r in pool}
    for filename in scan.stdout.splitlines():
        path = Path(filename)
        if path.name.startswith("candidates-"):
            continue
        found = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                found.update(regex.findall(line))
        for key in found:
            exposure[key].append(str(path.relative_to(ROOT)))
    for name, values in (("selected.jsonl", selected), ("pool.jsonl", pool), ("rejections.jsonl", rejected)):
        (folder / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")
    manifest = dict(kind="same-corpus supplementary evaluation; not an independently held-out benchmark",
        seed=20261006, questions=len(selected), source_counts=dict(Counter(r["source"] for r in selected)),
        eligible_pool=dict(Counter(r["source"] for r in pool)), rejection_counts=dict(Counter(r["reason"] for r in rejected)),
        corpus_sha256=cfg["corpus_sha256"], cases_sha256=sha(folder / "selected.jsonl"),
        selected_ids=[r["id"] for r in selected], historical_local_run_references=exposure,
        history_scan_scope="Local runs JSON/JSONL case IDs; does not prove absence of semantic or external exposure",
        selection_uses_model_or_retrieval_results=False, training=False,
        quality="Reused source structural checks, known semantic rejection ledger, exact evidence coverage; pending manual review")
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(folder=str(folder), eligible_pool=manifest["eligible_pool"],
        rejection_counts=manifest["rejection_counts"], selected_with_prior_references=sum(bool(exposure[r['id']]) for r in selected)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
