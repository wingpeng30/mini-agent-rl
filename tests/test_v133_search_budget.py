from __future__ import annotations

import asyncio
import json
from pathlib import Path
import uuid
from types import SimpleNamespace

from mini_agent_rl.domain import ToolCall
from mini_agent_rl.tools import SearchTool, ToolRegistry
from scripts.evaluate_v121_stop_control import visible_evidence
from scripts.evaluate_v133_search_budget import build_config, build_policy, execute_case, paired_summary, run_with_client, sha


class FixtureClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def reset_context_records(self):
        pass

    def context_usage_summary(self):
        return {"records": []}

    async def generate(self, messages, tools, temperature, seed):
        value = self.outputs.pop(0)
        return SimpleNamespace(
            tool_calls=value if isinstance(value, list) else [],
            content="" if isinstance(value, list) else value,
            raw_text="fixture",
            parse_error=None,
            usage={"prompt_sha256": "fixture", "prompt_tokens": 12},
            prompt_token_ids=[1, 2],
            prompt_attention_mask=[1, 1],
        )


class AnswerFixtureClient:
    def __init__(self, count):
        self.remaining = count
        self.calls = 0

    def reset_context_records(self):
        pass

    def context_usage_summary(self):
        return {"records": []}

    async def generate(self, messages, tools, temperature, seed):
        self.calls += 1
        self.remaining -= 1
        if self.remaining < 0:
            raise RuntimeError("fixture_call_budget_exhausted")
        return SimpleNamespace(tool_calls=[], content="answer", raw_text='{"action":"answer"}',
            parse_error=None, usage={"prompt_sha256": "fixture", "prompt_tokens": 12},
            prompt_token_ids=[1, 2], prompt_attention_mask=[1, 1])


class EmptyRegistry:
    def definitions(self):
        return []


def args(tmp_path: Path, tasks_path: Path, corpus_path: Path):
    return SimpleNamespace(tasks=str(tasks_path), corpus=str(corpus_path), model_path=str(tmp_path / "model"),
                           adapter_path=str(tmp_path / "adapter"), output_dir=str(tmp_path / "out"))


def registry():
    tools = ToolRegistry()
    tools.register(SearchTool([{"id": "d1", "title": "Alpha", "text": "Alpha fact."}], "bm25-v2"))
    return tools


def search(query):
    return [ToolCall(id="fixture", name="search", arguments={"query": query})]


def test_policy_changes_only_search_budget():
    a2, a3 = build_policy(2), build_policy(3)
    assert a2.rsplit(" Use at most ", 1)[0] == a3.rsplit(" Use at most ", 1)[0]
    assert a2.endswith("Use at most 2 searches total, including the first search.")
    assert a3.endswith("Use at most 3 searches total, including the first search.")


def test_search_budget_stops_and_counts_successful_calls():
    task = {"id": "q1", "question": "Who is Alpha?", "answer": "Alpha"}
    a2 = asyncio.run(execute_case(FixtureClient([search("Alpha"), search("Alpha fact"), search("third"), "Alpha"]),
                                 registry(), task, 2, 42))
    assert a2["actual_search_calls"] == 2
    assert a2["termination"] == "search_limit_insufficient"
    a3 = asyncio.run(execute_case(FixtureClient([search("Alpha"), search("Alpha fact"), search("third"), "Alpha"]),
                                 registry(), task, 3, 42))
    assert a3["actual_search_calls"] == 3
    assert a3["candidate_answer"] == "Alpha"


def test_repeat_gets_one_repair_and_repair_call_is_not_a_search():
    task = {"id": "q1", "question": "Who is Alpha?", "answer": "Alpha"}
    row = asyncio.run(execute_case(FixtureClient([search("Alpha"), search("Alpha"), search("Alpha fact"), "Alpha"]),
                                   registry(), task, 2, 42))
    assert row["actual_search_calls"] == 2
    assert row["repair_calls"] == 1
    assert row["termination"] == "answer"


def test_paired_summary_reports_transitions_and_zero_denominator_as_null():
    def row(task_id, em):
        return {"task_id": task_id, "normalized_exact_match": em, "token_f1": em,
                "actual_search_calls": 1, "protocol_valid": True}
    a2 = [row(f"q{i}", 0) for i in range(50)]
    a3 = [row(f"q{i}", 0) for i in range(50)]
    summary = paired_summary(a2, a3)
    assert summary["new_correct"] == summary["lost_correct"] == []
    assert summary["a2_correct_retention_rate"] is None


def test_truncated_visible_evidence_is_recorded_as_visible_excerpt():
    from mini_agent_rl.domain import Message
    messages = [Message(role="tool", content=json.dumps({"result": [{"id": "d1", "text": "abcdefgh"}]}))]
    context = {"document_changes": [{"visible": True, "deleted": False, "reused_reference": False,
                "tool_index": 0, "document_id": "d1", "title": "Alpha", "original_text": "abcdefgh",
                "content_sha256": "fixture", "visible_segments": [[1, 4]]}]}
    assert visible_evidence(messages, context) == [{"id": "d1", "title": "Alpha", "text": "bcd",
                                                   "content_sha256": "fixture", "visible_segments": [[1, 4]]}]


def test_each_row_is_saved_and_incomplete_arm_resumes():
    workspace = Path(__file__).resolve().parents[1]
    tmp_path = workspace / "runs" / f"v133-fixture-{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        tasks_path, corpus_path = tmp_path / "tasks.jsonl", tmp_path / "corpus.json"
        tasks = [{"id": f"q{i}", "question": f"Question {i}", "answer": "answer"} for i in range(50)]
        tasks_path.write_text("\n".join(json.dumps(row) for row in tasks), encoding="utf-8")
        corpus_path.write_text("[]", encoding="utf-8")
        run_args = args(tmp_path, tasks_path, corpus_path)

        interrupted = AnswerFixtureClient(1)
        try:
            asyncio.run(run_with_client(run_args, tasks, interrupted, EmptyRegistry()))
        except RuntimeError as exc:
            assert str(exc) == "fixture_call_budget_exhausted"
        else:
            raise AssertionError("fixture should stop after first durable row")
        first_arm = json.loads((tmp_path / "out/A2.json").read_text(encoding="utf-8"))
        assert len(first_arm["rows"]) == 1
        journal = [json.loads(line) for line in (tmp_path / "out/A2-events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert any(event["kind"] == "generation_result" and event["task_id"] == "q0" for event in journal)
        assert any(event["kind"] == "task_error" and event["task_id"] == "q1" for event in journal)
        try:
            asyncio.run(run_with_client(run_args, tasks, AnswerFixtureClient(99), EmptyRegistry()))
        except ValueError as exc:
            assert str(exc) == "partial_task_attempt_preserved_and_requires_manual_recovery"
        else:
            raise AssertionError("incomplete task must not be silently retried")

        safe_args = args(tmp_path, tasks_path, corpus_path)
        safe_args.output_dir = str(tmp_path / "safe-out")
        safe_out = Path(safe_args.output_dir)
        safe_out.mkdir()
        safe_settings = build_config(safe_args, tasks)
        settings_path = safe_out / "settings.json"
        settings_path.write_text(json.dumps(safe_settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        seed_row = {"task_id": "q0", "normalized_exact_match": 1.0, "token_f1": 1.0,
                    "actual_search_calls": 0, "protocol_valid": True}
        (safe_out / "A2.json").write_text(json.dumps({"version": "v133-search-budget-arm-v1", "arm": "A2",
            "settings_sha256": sha(settings_path), "rows": [seed_row]}), encoding="utf-8")

        resume = AnswerFixtureClient(99)
        summary = asyncio.run(run_with_client(safe_args, tasks, resume, EmptyRegistry()))
        assert resume.calls == 99
        assert summary["n"] == 50
        assert json.loads((safe_out / "A2.json").read_text(encoding="utf-8"))["complete_coverage"]
        assert json.loads((safe_out / "A3.json").read_text(encoding="utf-8"))["complete_coverage"]
    finally:
        for path in (tmp_path / "tasks.jsonl", tmp_path / "corpus.json", tmp_path / "out/settings.json",
                     tmp_path / "out/A2.json", tmp_path / "out/A3.json", tmp_path / "out/paired-summary.json",
                     tmp_path / "out/A2-events.jsonl", tmp_path / "out/A3-events.jsonl",
                     tmp_path / "safe-out/settings.json", tmp_path / "safe-out/A2.json",
                     tmp_path / "safe-out/A3.json", tmp_path / "safe-out/paired-summary.json",
                     tmp_path / "safe-out/A2-events.jsonl", tmp_path / "safe-out/A3-events.jsonl"):
            if path.is_file():
                path.unlink()
        for out in (tmp_path / "out", tmp_path / "safe-out"):
            if out.is_dir():
                out.rmdir()
        tmp_path.rmdir()
