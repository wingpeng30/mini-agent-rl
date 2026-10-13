import json


def test_reverse_changes_only_document_order():
    import run_v133_goal_r_evidence_order as r
    observation = {"ok": True, "result": [{"id": "a", "text": "one", "score": 2},
                                                 {"id": "b", "text": "two", "score": 1}],
                   "searched_queries": ["query"]}
    trace = [{"action": "search", "query": "query", "observation": observation},
             {"action": "answer", "answer": "value"}]
    original = r.rebuild("Question?", trace, reverse=False)
    reversed_ = r.rebuild("Question?", trace, reverse=True)
    assert [(x.role, x.content) for x in original[:3]] == [(x.role, x.content) for x in reversed_[:3]]
    first = json.loads(original[-1].content)
    second = json.loads(reversed_[-1].content)
    assert first["result"] == list(reversed(second["result"]))
    assert {k: v for k, v in first.items() if k != "result"} == {k: v for k, v in second.items() if k != "result"}
    assert observation["result"][0]["id"] == "a"


def test_selection_requires_gain_and_regression(monkeypatch):
    import run_v133_goal_r_evidence_order as r
    captured = {}
    monkeypatch.setattr(r, "write", lambda path, item: captured.update(item))
    values = {"incumbent": (19, 7, 0, 32), "a": (21, 6, 0, 33), "b": (20, 7, 0, 35)}
    monkeypatch.setattr(r, "stats", lambda label: {
        "model": label, "total": values[label][0], "regression": values[label][1],
        "invalid": values[label][2], "searches": values[label][3]})
    assert r.select() == "b"
    assert len(captured["eligible"]) == 1


def test_answer_target_contains_only_final_action():
    import run_v133_goal_r_evidence_order as r
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(text.encode("utf-8"))
        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 999
    ids, raw = r.answer_target(type("Processor", (), {"tokenizer": Tokenizer()})(), "writer")
    assert json.loads(raw) == {"action": "answer", "answer": "writer"}
    assert ids[-1] == 999
    assert bytes(ids[:-1]).decode("utf-8") == raw


def test_persisted_document_fields_restore_runtime_order():
    from repair_v133_goal_r_diagnostic import runtime_order
    old = [{"observation": {"result": [{"id": "a", "score": 1.0,
                                           "text": "evidence", "title": "A"}]}}]
    new = runtime_order(old)
    assert list(new[0]["observation"]["result"][0]) == ["id", "title", "text", "score"]
    assert old[0]["observation"]["result"][0]["score"] == 1.0
