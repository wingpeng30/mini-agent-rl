"""Audit whether the 35 unpaired CPU diagnostics are independent trajectories."""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.v123_cpu_core import ROOT, readl, write
from scripts.v124_forensic import OUT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_path = ROOT / "scripts/build_v114_sft_candidates.py"
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    function = next(node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "build_examples")
    names = Counter(node.id for node in ast.walk(function) if isinstance(node, ast.Name))
    diagnostic_path = OUT / "remaining-unpaired-diagnostic-private.jsonl"
    triage_paths = [
        OUT / "v122-train-unpaired-semantic-triage-private.jsonl",
        OUT / "v123-unpaired-semantic-triage-private.jsonl",
    ]
    diagnostic = readl(diagnostic_path)
    triage = [row for path in triage_paths for row in readl(path)]
    if len(diagnostic) != 35 or len(triage) != 35:
        raise ValueError("unpaired_record_count_changed")
    if {r["source_id"] for r in diagnostic} != {r["source_id"] for r in triage}:
        raise ValueError("diagnostic_triage_binding_changed")

    required_dependencies = {"support_facts", "facts_visible"}
    if not required_dependencies <= set(names):
        raise RuntimeError("expected_gold_fact_dependency_not_found")
    output = {
        "version": "v126-unpaired-oracle-construction-audit-v1",
        "inputs": {
            str(source_path.relative_to(ROOT)): sha(source_path),
            str(diagnostic_path.relative_to(ROOT)): sha(diagnostic_path),
            **{str(p.relative_to(ROOT)): sha(p) for p in triage_paths},
        },
        "diagnostic_questions": len(diagnostic),
        "triage_dispositions": dict(Counter(r["disposition"] for r in triage)),
        "builder_function_ast_name_counts": {
            name: names[name] for name in sorted(required_dependencies | {
                "question_queries", "matched", "partial_runs", "full_runs",
                "missing_facts", "closed", "query_options", "facts_visible"})
        },
        "code_path_findings": [
            "build_examples obtains required_facts from the row's supporting-fact annotations.",
            "It selects question-derived initial queries only after checking their retrievals against those required facts (matched, partial_runs, full_runs).",
            "It selects later queries using whether new retrievals close annotated missing_facts, then emits supported_answer only when facts_visible(required_facts, retrieved) is true.",
            "The unpaired diagnostic invokes build_examples and records state counts, not persisted, independently reviewed model-generated action trajectories.",
        ],
        "interpretation": {
            "classification": "gold-conditioned synthetic construction; valid only as a structural upper-bound diagnostic",
            "may_be_counted_as_model_generated_trajectory": False,
            "may_be_exported_as_independent_search_action_label": False,
            "review_status": "not_independently_semantically_reviewed_as_new_trajectories",
            "training_allowed": False,
        },
        "training_allowed": False,
    }
    path = OUT / "v126-unpaired-oracle-construction-audit-private.json"
    if path.exists():
        raise FileExistsError(path)
    write(path, output)
    print(json.dumps({
        "version": output["version"],
        "diagnostic_questions": output["diagnostic_questions"],
        "triage_dispositions": output["triage_dispositions"],
        "gold_fact_dependencies_present": True,
        "independent_trajectories": False,
        "training_allowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
