import json

from mini_agent_rl.training.audit import audit_sft_directory, audit_final_test


def test_audit_detects_document_overlap(tmp_path):
    for split, title in (("train", "甲"), ("validation", "乙"), ("test", "甲")):
        record = {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": split}], "metadata": {"search_count": 1, "support_titles": [title]}}
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    report = audit_sft_directory(tmp_path)
    assert report["document_overlap"]["train_test"] == 1
    assert report["passed_document_isolation"] is False


def test_final_test_audit_checks_isolation_and_manifest(tmp_path):
    """最终集审计必须同时验证 task、supporting document 与内容哈希。"""
    source = tmp_path / "source"; final = tmp_path / "final"
    source.mkdir(); final.mkdir()
    for split in ("train", "validation", "test"):
        record = {"messages": [], "metadata": {"task_id": f"old-{split}", "support_titles": [f"标题-{split}"]}}
        (source / f"{split}.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    record = {"messages": [], "metadata": {"task_id": "new-1", "support_titles": ["新标题"]}}
    final_path = final / "final-test.jsonl"
    final_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    from mini_agent_rl.training.hotpot import _file_sha256
    (final / "manifest.json").write_text(json.dumps({"final_test": {"sha256": _file_sha256(final_path)}}), encoding="utf-8")
    assert audit_final_test(source, final)["passed"] is True
