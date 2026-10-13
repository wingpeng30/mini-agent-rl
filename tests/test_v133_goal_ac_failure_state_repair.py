import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_v133_goal_ac_failure_state_repair as ac
import run_v133_goal_ac_supplement as supplement
import continue_v133_goal_ac_replay_fix as replay
import demo_v133_goal_z_selected as demo


def test_replay_rebuilds_live_tool_document_key_order(monkeypatch):
    monkeypatch.setattr(ac.q, "load_config", lambda: {"product": {"system_prompt": "direct-v090"}})
    trace = [{"action": "search", "query": "Vilsandi", "observation": {
        "ok": True, "result": [{"id": "d1", "score": 4.2, "text": "Located in the Baltic Sea.",
                                 "title": "Vilsandi"}], "searched_queries": ["vilsandi"]}}]
    messages = replay.product_order_messages_for({"question": "Which sea contains Vilsandi?",
        "trace": trace}, 1, None)
    payload = json.loads(messages[-1].content)
    assert list(payload["result"][0]) == ["id", "title", "text", "score"]


def test_action_tokenizer_adds_message_terminator():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return [11, 12]

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 99

    raw, ids = replay.exact_action_ids(Tokenizer(), "answer", answer="A")
    assert json.loads(raw) == {"action": "answer", "answer": "A"}
    assert ids == [11, 12, 99]


def test_disagreeing_supplemental_review_is_not_repairable(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "OUT", tmp_path)
    packet = [{"case_id": "case-1"}]
    ac.write_jsonl(tmp_path / "extra-failure-audit-packet.jsonl", packet)
    for name, event in (("extra-audit-a.jsonl", "premature_answer"),
                        ("extra-audit-b.jsonl", "wrong_bridge_query")):
        ac.write_jsonl(tmp_path / name, [{"case_id": "case-1", "event": event,
            "error_step": 1, "repairable_within_budget": True, "error_supported": True,
            "reason": "independent review", "repair_queries": [{"query": "grounded query",
            "bridge_source": "entity in question"}]}])
    result = supplement.combine_extra_audits()
    vote = ac.rows(tmp_path / "extra-failure-audit-final.jsonl")[0]
    assert result["repairable"] == 0
    assert vote["event"] == "ambiguous"
    assert vote["repairable_within_budget"] is False


def test_product_entry_uses_published_ac_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.z, "ROOT", tmp_path)
    out = tmp_path / "runs/v133-goal-ac-failure-state-repair"
    out.mkdir(parents=True)
    adapter = str(out / "seed42-C/adapter-final")
    (out / "product-decision.json").write_text(json.dumps({"published": True,
        "status": "published_C42", "product_adapter": adapter}), encoding="utf-8")
    (out / "state.json").write_text(json.dumps({"status": "published_C42"}), encoding="utf-8")
    assert demo.selected_adapter() == ("Goal AC failure-state repair seed 42", adapter)
