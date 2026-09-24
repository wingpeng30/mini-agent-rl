import json
from pathlib import Path

from mini_agent_rl.evaluation import load_hotpot_environment
from mini_agent_rl.cli import _retrieval_fingerprint
from mini_agent_rl.tools import SearchTool


def test_task_limit_does_not_change_fixed_corpus_hash():
    path = Path(__file__).parents[1] / "data/hotpot-agent-v041/train.jsonl"
    small = load_hotpot_environment(path, task_limit=5, corpus_limit=100)
    large = load_hotpot_environment(path, task_limit=10, corpus_limit=100)
    assert small.corpus_sha256 == large.corpus_sha256
    assert small.task_ids == large.task_ids[:5]
    assert _retrieval_fingerprint(small, "bm25-v2", "compact-v3") == _retrieval_fingerprint(large, "bm25-v2", "compact-v3")


def test_bm25_v2_filters_single_latin_tokens_and_keeps_chinese():
    assert "s" not in __import__("mini_agent_rl.tools.search", fromlist=["_v2_tokens"])._v2_tokens("Arthur's Magazine")
    tokens = __import__("mini_agent_rl.tools.search", fromlist=["_v2_tokens"])._v2_tokens("故宫 位于 北京")
    assert "故" in tokens and "宫" in tokens and "北" in tokens


def test_bm25_v2_exact_title_is_deterministic_and_top_ranked():
    corpus = [
        {"id": "b", "title": "House (TV series)", "text": "Arthur magazine history and television."},
        {"id": "a", "title": "Arthur's Magazine", "text": "An American literary periodical."},
    ]
    tool = SearchTool(corpus, "bm25-v2")
    first = tool._search("Arthur's Magazine")
    second = tool._search("Arthur's Magazine")
    assert first == second
    assert first[0]["title"] == "Arthur's Magazine"


def test_search_tool_does_not_consume_task_answer_metadata():
    corpus = [{"id": "1", "title": "A", "text": "evidence"}]
    tool = SearchTool(corpus, "bm25-v2")
    assert "answer" not in tool.corpus[0]
