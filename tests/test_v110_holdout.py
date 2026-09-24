import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from mini_agent_rl.evaluation import load_hotpot_environment

spec = importlib.util.spec_from_file_location("prepare_v110", Path(__file__).parents[1] / "scripts/prepare_v110_holdout.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source_row(identifier, question=None):
    return {"_id": identifier, "question": question or f"Question {identifier}?", "answer": "gold",
            "context": [["Support", ["same text"]], ["Distractor", ["unrelated"]], ["Variant", [identifier]]],
            "supporting_facts": [["Support", 0]]}


def test_exclusions_order_full_context_and_loader(tmp_path):
    rows = [source_row(str(i)) for i in range(8)] + [source_row("overlap", " KNOWN， question! ")]
    source = tmp_path / "dev.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    exclude = tmp_path / "train.jsonl"
    exclude.write_text(json.dumps({"metadata": {"task_id": "hotpot-0"}, "messages": [{"role": "user", "content": "Known question"}]}), encoding="utf-8")
    output = tmp_path / "out"
    manifest = module.prepare(source, [exclude], output, 3)
    expected = sorted([str(i) for i in range(1, 8)], key=lambda x: hashlib.sha256(("v110:" + x).encode()).hexdigest())[:3]
    assert manifest["task_ids"] == ["hotpot-" + x for x in expected]
    assert manifest["excluded_rows"] == 2
    small = load_hotpot_environment(output / "holdout.jsonl", task_limit=1, corpus_limit=3)
    full = load_hotpot_environment(output / "holdout.jsonl", task_limit=3, corpus_limit=3)
    assert small.corpus_sha256 == full.corpus_sha256 == manifest["files"]["corpus.json"]
    assert len(full.corpus) == 5  # exact repeats merge, same-title text variants survive
    assert "Distractor" in {d["title"] for d in full.corpus}
    assert full.tasks[0].metadata["support_titles"] == ["Support"]
    source.write_text(json.dumps(list(reversed(rows))), encoding="utf-8")
    second = module.prepare(source, [exclude], tmp_path / "second", 3)
    assert second["files"] == manifest["files"]


def test_final_test_rejected_before_read(tmp_path):
    with pytest.raises(ValueError, match="final-test"):
        module.prepare(tmp_path / "dev.json", [tmp_path / "final-test.jsonl"], tmp_path / "out")


def test_cross_component_path_is_allowed(tmp_path):
    assert module._safe_path(tmp_path / "preflight-final" / "test-data")


def test_parquet_canonical_shape(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    row = source_row("a")
    row["id"] = row.pop("_id")
    row["context"] = {"title": [x[0] for x in row["context"]], "sentences": [x[1] for x in row["context"]]}
    row["supporting_facts"] = {"title": ["Support"], "sent_id": [0]}
    source = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist([row]), source)
    assert module._rows(source)[0]["context"] == [("Support", ["same text"]), ("Distractor", ["unrelated"]), ("Variant", ["a"])]


def test_corpus_does_not_depend_on_supporting_fact_labels(tmp_path):
    row = source_row("a")
    source = tmp_path / "dev.json"
    source.write_text(json.dumps([row]), encoding="utf-8")
    exclude = tmp_path / "train.jsonl"
    exclude.write_text("", encoding="utf-8")
    first = module.prepare(source, [exclude], tmp_path / "first", 1)
    row["supporting_facts"] = [["Distractor", 0]]
    source.write_text(json.dumps([row]), encoding="utf-8")
    second = module.prepare(source, [exclude], tmp_path / "second", 1)
    assert first["files"]["corpus.json"] == second["files"]["corpus.json"]


def test_insufficient_rows_no_partial_output(tmp_path):
    source = tmp_path / "dev.json"
    source.write_text(json.dumps([source_row("a")]), encoding="utf-8")
    exclude = tmp_path / "train.jsonl"
    exclude.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="require 100"):
        module.prepare(source, [exclude], tmp_path / "out")
    assert not (tmp_path / "out").exists()
