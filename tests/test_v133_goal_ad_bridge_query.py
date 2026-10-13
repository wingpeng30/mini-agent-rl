import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_v133_goal_ad_bridge_query as ad


def test_visible_document_reader_keeps_only_prompt_fragment_and_source_span():
    prompt = (
        "<|im_start|>user\n<tool_response>{\"ok\":true,\"result\":["
        "{\"id\":\"d7\",\"title\":\"A title\",\"text\":\"visible clipped tail\"}]}</tool_response>"
    )
    docs = ad.visible_documents(prompt)
    assert len(docs) == 1
    assert docs[0]["id"] == "d7"
    assert docs[0]["text"] == "visible clipped tail"
    assert docs[0]["fragment_complete"] is True
    assert prompt[docs[0]["text_char_span"][0]:docs[0]["text_char_span"][1]] == "visible clipped tail"
    assert "unseen beginning" not in docs[0]["text"]


def test_non_tool_history_text_is_not_counted_as_visible_evidence():
    prompt = (
        "<|im_start|>user\nQuestion mentions document d1.\n"
        "<tool_response>{\"ok\":true,\"result\":[{\"id\":\"d2\",\"title\":\"T\","
        "\"text\":\"only returned passage\"}]}</tool_response>"
    )
    assert [d["id"] for d in ad.visible_documents(prompt)] == ["d2"]


def test_review_ids_must_cover_every_case_once():
    packet = [{"case_id": "a"}, {"case_id": "b"}]
    assert set(ad.check_id_rows(packet, [{"case_id": "b"}, {"case_id": "a"}])) == {"a", "b"}
    with pytest.raises(RuntimeError):
        ad.check_id_rows(packet, [{"case_id": "a"}, {"case_id": "a"}])


def test_query_author_proposal_must_ground_bridge_in_input_and_query():
    packet = [{"case_id": "x", "question": "Which award did Liora's director win?",
        "actual_insufficient_prompt": "director is Mira Vale", "searches_before_error": 1,
        "insufficient_visible_documents": []}]
    import hashlib
    author = [{"case_id": "x", "input_sha256": hashlib.sha256(
            (packet[0]["question"] + "\n" + packet[0]["actual_insufficient_prompt"]).encode()).hexdigest(),
        "question": packet[0]["question"], "actual_insufficient_prompt": packet[0]["actual_insufficient_prompt"],
        "searches_before_error": 1}]
    path = {"x": {"case_id": "x", "training_answer": "Aurora Prize"}}
    diagnosis = {"x": {"case_id": "x", "query_remediable": True}}
    valid_proposal = [{"case_id": "x", "input_sha256": author[0]["input_sha256"],
        "queries": [{"query": "Mira Vale award", "bridge_source": "Mira Vale"}]}]
    assert ad.validate_query_proposal_rows(author, valid_proposal, path, diagnosis)[1]["eligible_cases"] == 1
    invalid_proposal = [{"case_id": "x", "input_sha256": author[0]["input_sha256"],
        "queries": [{"query": "award ceremony winner", "bridge_source": "Mira Vale"}]}]
    with pytest.raises(RuntimeError, match="does not use its cited bridge entity"):
        ad.validate_query_proposal_rows(author, invalid_proposal, path, diagnosis)


def test_exact_query_token_builder_uses_product_message_terminator():
    class FakeTokenizer:
        def encode(self, raw, add_special_tokens=False):
            assert json.loads(raw) == {"action": "search", "query": "Mira Vale"}
            return [11, 12]

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 13

    raw, tokens = ad.exact_search_tokens(FakeTokenizer(), "Mira Vale")
    assert json.loads(raw)["query"] == "Mira Vale"
    assert tokens == [11, 12, 13]


def test_repair_corpus_uses_frozen_case_membership_not_source_label():
    goal_z_ids = {"musique:2hop__68170_383079", "old:other"}
    assert ad.repair_corpus_key("musique:2hop__68170_383079", goal_z_ids) == "goal_z_train"
    assert ad.repair_corpus_key("2wiki:extra-case", goal_z_ids) == "ac_extra_train"


def test_runner_hash_refresh_is_idempotent_after_reviews_are_submitted(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    state_path, cfg_path = out / "state.json", out / "config.json"
    source = out / "runner.py"
    source.write_text("frozen runner", encoding="utf-8")
    (out / "root-path-adjudication.jsonl").write_text("reviewed", encoding="utf-8")
    monkeypatch.setattr(ad, "OUT", out)
    monkeypatch.setattr(ad, "STATE", state_path)
    monkeypatch.setattr(ad, "CFG", cfg_path)
    monkeypatch.setattr(ad, "sources", lambda: {"ad_runner": source})
    ad.write(state_path, {"status": "awaiting_path_adjudication"})
    ad.write(cfg_path, {"source_hashes": {"ad_runner": ad.sha(source)}})

    ad.refresh_pretraining_runner_hash()

    assert ad.read(cfg_path)["source_hashes"]["ad_runner"] == ad.sha(source)
