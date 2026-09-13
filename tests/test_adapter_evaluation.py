import json

import pytest

from mini_agent_rl.domain import Rollout, RolloutStatus, Task, Transition
from mini_agent_rl.evaluation import evaluate_tasks, hotpot_sft_to_tasks_and_corpus


def test_hotpot_sft_conversion_keeps_gold_answer_out_of_corpus(tmp_path):
    record = {
        "messages": [
            {"role": "system", "content": "协议"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": '{"action":"search","query":"文档"}'},
            {"role": "tool", "content": json.dumps({"ok": True, "result": [{"id": "d1", "title": "文档", "text": "证据"}]})},
            {"role": "assistant", "content": '{"action":"answer","answer":"答案"}'},
        ],
        "metadata": {"task_id": "t1", "gold_answer": "答案", "source": "test"},
    }
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(path)
    assert tasks[0].answer == "答案"
    assert corpus == [{"id": "d1", "title": "文档", "text": "证据"}]
    assert all("答案" not in json.dumps(item, ensure_ascii=False) for item in corpus)


@pytest.mark.asyncio
async def test_evaluation_counts_parse_failure_as_invalid_action():
    class Store:
        def get_rollout(self, rollout_id):
            return Rollout(id=rollout_id, group_id="g", task_id="t", seed=0, status=RolloutStatus.FAILED, error="模型输出不是合法 JSON: Expecting value")
        def get_transitions(self, rollout_id):
            return []
    class Group:
        rollout_ids = ["r"]
    class Runner:
        store = Store()
        model = object()
        async def run_group(self, task, group_size):
            return Group()
    report = await evaluate_tasks(Runner(), [Task(id="t", question="q", answer="a")], 1)
    assert report["invalid_action_rate"] == 1.0
    assert report["valid_action_rate"] == 0.0
