import json

from mini_agent_rl.training.hotpot import _record, _split_for_titles, _support_docs


def test_hotpot_record_has_two_searches_and_answer():
    example = {
        "id": "example", "question": "问题？", "answer": "答案", "level": "medium",
        "supporting_facts": {"title": ["甲", "乙"], "sent_id": [0, 0]},
        "context": {"title": ["甲", "乙"], "sentences": [["甲的内容"], ["乙的内容"]]},
    }
    docs = _support_docs(example)
    record = _record(example, docs, "train")
    assert record["metadata"]["search_count"] == 2
    assert json.loads(record["messages"][2]["content"])["action"] == "search"
    assert json.loads(record["messages"][-1]["content"])["answer"] == "答案"


def test_document_split_is_deterministic():
    assert _split_for_titles(["固定标题", "固定标题"]) == _split_for_titles(["固定标题", "固定标题"])
    assert _split_for_titles(["甲", "乙", "丙"]) in {None, "train", "validation", "test"}
