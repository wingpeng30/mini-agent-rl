import json
import asyncio
from types import SimpleNamespace

import pytest

from scripts.evaluate_v122_unified import ARM_PROMPTS, choose, grounded_query, require_preflight, select_preflight
from mini_agent_rl.domain import Message, ToolCall
from scripts import evaluate_v122_unified as v122
from scripts.audit_v122_api import packets
from scripts.merge_v122_audit import review_for
from scripts.audit_v121_arms_api import digest


def test_all_arms_keep_the_trained_action_protocol():
    for prompt in ARM_PROMPTS.values():
        assert "JSON" in prompt
        assert "search or answer action" in prompt
        assert '"status"' not in prompt
        assert '"facts"' not in prompt


def test_query_anchor_must_be_present_before_search():
    evidence = [{"id": "d1", "text": "Kevin Rathbun ran restaurants in Atlanta."}]
    assert grounded_query("Atlanta population 2016", "What was the city's population in 2016?", evidence)
    assert not grounded_query("Rathbun Iowa population 2016", "What was the city's population in 2016?", evidence)
    assert not grounded_query("2016 population", "What was the city's population in 2016?", evidence)


def test_preflight_excludes_development_question(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = []
    for kind in ("search", "answer"):
        for i in range(26):
            target = (json.dumps({"action":"search","query":f"Entity{i} birth date"}) if kind == "search"
                      else json.dumps({"action":"answer","answer":f"Entity{i}"}))
            rows.append({"messages":[{"role":"system","content":"Act."},
                                     {"role":"user","content":f"Question {kind} {i}?"},
                                     {"role":"assistant","content":target}],
                         "target_assistant_index":2})
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    selected = select_preflight(path, {"question search 0?", "question answer 0?"})
    assert {kind:len(states) for kind,states in selected.items()} == {"search":25,"answer":25}
    assert all(" 0?" not in states[1].content for group in selected.values() for states in group)


def test_candidate_requires_all_preregistered_gates():
    def report(correct, preserve=14, f1=.7, protocol=1., searches=90):
        return {"metrics":{"n":50,"correct":correct,"mean_f1":f1,"preserve_v118":preserve,
                           "preserve_v120":25,"protocol_rate":protocol,"searches":searches}}
    reports = {"A":report(28,12,.6),"B":report(31,14),"C":report(32,13),"D":report(30,14)}
    assert choose(reports) == "B"
    reports["B"] = report(31,14,protocol=.96)
    assert choose(reports) is None


def test_changed_settings_or_missing_probe_blocks_development_run(tmp_path):
    with pytest.raises(ValueError, match="passing_matching_preflight_required"):
        require_preflight(tmp_path, "expected")
    gate = {"version":"v122-preflight-v1","settings_sha256":"old","counts":{arm:50 for arm in "ABCD"},"passed":True}
    (tmp_path / "preflight-report.json").write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="passing_matching_preflight_required"):
        require_preflight(tmp_path, "expected")
    gate["settings_sha256"] = "expected"
    (tmp_path / "preflight-report.json").write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="probe_missing_A"):
        require_preflight(tmp_path, "expected")


def test_repeat_repair_uses_old_action_format_and_future_evidence_stays_future(monkeypatch):
    seen_messages = []
    outputs = [
        ("search", "Atlanta"),
        ("search", "Atlanta population 2016"),
        ("answer", "500000"),
    ]

    async def fake_generate(client, messages, tools, seed):
        seen_messages.append([m.content for m in messages])
        kind, text = outputs.pop(0)
        response = SimpleNamespace(content=text if kind == "answer" else "",
                                   tool_calls=[ToolCall(name="search",arguments={"query":text})] if kind == "search" else [])
        return response, {"action":kind,"visible_evidence":[{"id":"d1","text":"Kevin Rathbun worked in Atlanta."}]}

    class Registry:
        def __init__(self):
            self.queries = []
        def definitions(self):
            return []
        async def execute(self, name, arguments):
            self.queries.append(arguments["query"])
            return {"ok":True,"result":[{"id":f"doc-{len(self.queries)}",
                                          "text":"Atlanta population was 500000 in 2016." if len(self.queries) == 2 else "Kevin Rathbun worked in Atlanta."}]}

    monkeypatch.setattr(v122, "generate", fake_generate)
    registry = Registry()
    source = [Message(role="system",content="Return one JSON action."),
              Message(role="user",content="What was the population of Rathbun's city in 2016?")]
    result = asyncio.run(v122.run_case(None, registry, "A", source, source[1].content, "Atlanta", 42))
    assert registry.queries == ["Atlanta", "Atlanta population 2016"]
    assert result["repair_used"] is True and result["candidate_answer"] == "500000"
    assert "500000" not in " ".join(seen_messages[0])
    assert "500000" in " ".join(seen_messages[-1])


def test_api_packets_hide_scores_gold_and_arm_names(tmp_path):
    tasks = {f"q{i}":{"question":f"Who is person {i}?", "answer":f"secret {i}"} for i in range(50)}
    for arm in "ABCD":
        rows = [{"task_id":task_id,"calls":[{"role":"policy_action","action":"answer",
                 "raw_output":'{"action":"answer","answer":"Person"}',"visible_evidence":[]}],
                 "retrieved_docs":[],"candidate_answer":"Person","candidate_em":0,
                 "candidate_f1":0} for task_id in tasks]
        (tmp_path / f"fixed-{arm}.json").write_text(json.dumps({"settings_sha256":"same",
            "complete_coverage":True,"rows":rows}),encoding="utf-8")
    result = packets(tmp_path,tasks)
    assert len(result) == 200
    transmitted = json.dumps(result)
    assert "secret" not in transmitted
    assert '"task_id"' not in transmitted and '"candidate_em"' not in transmitted
    assert '"arm"' not in transmitted


def test_blind_review_cannot_quote_future_retrieval_as_prior_support():
    packet = {"case_id":"opaque","traces":[{"trace_id":"trace-x","decisions":[
        {"decision_index":1,"turn_purpose":"action_proposal","action":"answer",
         "visible_evidence":[{"visible_text":"Kevin worked in Atlanta."}]}],
        "searches":[{"search_index":1,"top_results":[{"retrieved_text":"Atlanta had 500000 people."}],
                     "evidence_after":[]}]}]}
    review = {"case_id":"opaque","packet_sha256":digest(json.dumps(packet,ensure_ascii=False,sort_keys=True)),
              "citation_errors":[],"result":{"cases":[{"case_id":"opaque","traces":[{
              "trace_id":"trace-x","final_answer_supported":"yes",
              "decision_reviews":[{"decision_index":1,"answer_supported":"yes",
                                   "support_quotes":["Atlanta had 500000 people."]}],
              "search_reviews":[{"search_index":1,"quotes":[]}]}]}]}}
    result,errors = review_for(packet,review)
    assert result is None and "invalid_decision_quote" in errors


def test_new_internal_selection_requires_50_pairs_and_risk_coverage():
    from scripts.finalize_v122_internal import choose
    plain = [(f"q{i:03d}", [(None, []), (None, [])]) for i in range(60)]
    assert choose(plain[:49]) is None
    assert choose(plain) is None
    covered = [(f"q{i:03d}", [(None, ["relation", "time", "scope", "measurement"]),
                               (None, ["relation", "time", "scope", "measurement"])])
               for i in range(60)]
    selected = choose(covered)
    assert selected is not None and [qid for qid, _ in selected] == [f"q{i:03d}" for i in range(50)]


def test_targeted_train_selection_requires_relation_support():
    from scripts.finalize_v122_train_supplement import choose_relation_pairs
    plain = [(f"q{i:03d}", [(None, []), (None, [])]) for i in range(30)]
    assert choose_relation_pairs(plain) is None
    enriched = plain[:10] + [(f"q{i:03d}", [(None, ["relation"]), (None, ["relation"])])
                             for i in range(10, 40)]
    selected = choose_relation_pairs(enriched)
    assert selected is not None and len(selected) == 25
    assert sum("relation" in labels for _, records in selected for _, labels in records) >= 20


def test_v122_training_cannot_start_with_unapproved_data(tmp_path):
    from scripts.train_v122 import check
    from types import SimpleNamespace
    paths = {}
    for name, value in {"remote_preflight": {"version": "v122-remote-preflight-v1", "training_permitted": True},
                        "token_audit": {}, "train_gate": {"local_train_ready": False},
                        "internal_gate": {"local_internal_ready": True}}.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    args = SimpleNamespace(**paths)
    with pytest.raises(ValueError, match="passing_data_and_remote_preflight_required"):
        check(args)


def test_v122_training_rejects_changed_frozen_config(tmp_path):
    from scripts.train_v122 import check
    files = {}
    for name, value in {
        "remote_preflight": {"version": "v122-remote-preflight-v1", "training_permitted": True,
                             "training_config_sha256": "stale"},
        "token_audit": {}, "train_gate": {"local_train_ready": True},
        "internal_gate": {"local_internal_ready": True}, "config": {"version": "changed"},
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        files[name] = path
    with pytest.raises(ValueError, match="frozen_training_configuration_changed"):
        check(SimpleNamespace(**files))


def test_new_internal_audit_uses_only_truncated_action_time_evidence():
    from scripts.evaluate_v122_internal import actual_input
    messages = [{"role": "user", "content": "Which city?"},
                {"role": "tool", "content": json.dumps({"result": [
                    {"id": "d1", "title": "City", "text": "visible start. hidden ending."}]})}]
    response = {"prompt_token_ids": [1, 2], "prompt_attention_mask": [1, 1],
                "usage": {"prompt_sha256": "abc", "document_changes": [
                    {"visible": True, "deleted": False, "reused_reference": False,
                     "tool_index": 0, "document_id": "d1", "title": "City",
                     "original_text": "visible start. hidden ending.", "visible_segments": [[0, 14]]}]}}
    result = actual_input(messages, response)
    assert result["visible_evidence"][0]["text"] == "visible start."
    assert "hidden ending" not in json.dumps(result)


def test_blind_packet_builder_excludes_untruncated_evidence():
    from scripts.build_v122_internal_audit import action_time_visible_messages
    actual = {"prompt_token_ids": [1, 2], "prompt_sha256": "prompt-hash",
              "visible_evidence": [{"id": "d1", "text": "Visible sentence."}]}
    messages = action_time_visible_messages("Who?", actual)
    assert "Visible sentence." in json.dumps(messages)
    assert "prompt_token_ids" not in json.dumps(messages)
    assert "gold" not in json.dumps(messages)
    with pytest.raises(ValueError, match="missing_actual_input_binding"):
        action_time_visible_messages("Who?", {"visible_evidence": []})


def test_new_internal_gate_requires_every_registered_metric(tmp_path):
    from scripts.gate_v122_internal import run
    paths = {}
    for label in ("v101", "v118", "v122"):
        for mode in ("states", "tasks"):
            if mode == "states":
                rows = [{"state_id": f"s{i}", "target": {"action": "answer" if i < 50 else "search"},
                         "protocol_valid": True, "normalized_target_match": True, "action_match": True}
                        for i in range(100)]
            else:
                rows = [{"task_id": f"q{i}", "normalized_exact_match": i < (20 if label == "v101" else 23),
                         "token_f1": 1.0 if i < (20 if label == "v101" else 23) else 0.0,
                         "protocol_valid": True} for i in range(50)]
            report = {"version": "v122-new-internal", "model_label": label, "mode": mode,
                      "complete_coverage": True, "rows": rows, "corpus_sha256": "frozen",
                      "runner_config": {}, "code_sha256": {}, "model_files": {}}
            path = tmp_path / f"{label}-{mode}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            paths[f"{label}_{mode}"] = path
    audit = tmp_path / "semantic.json"
    audit.write_text(json.dumps({"version": "v122-semantic-gate-v1", "state_cases": 100,
                                  "semantic_gate_passed": True, "unresolved_disputes": 0,
                                  "confirmed_semantic_regressions": 0}), encoding="utf-8")
    result = run(paths, audit, tmp_path / "gate.json")
    assert result["internal_passed"] is True
    changed = json.loads(paths["v122_states"].read_text(encoding="utf-8"))
    changed["rows"][0]["normalized_target_match"] = False
    changed["rows"][1]["normalized_target_match"] = False
    changed["rows"][2]["normalized_target_match"] = False
    changed["rows"][3]["normalized_target_match"] = False
    changed["rows"][4]["normalized_target_match"] = False
    changed["rows"][5]["normalized_target_match"] = False
    paths["v122_states"].write_text(json.dumps(changed), encoding="utf-8")
    assert run(paths, audit, tmp_path / "gate2.json")["internal_passed"] is False


def test_remote_runner_smoke_and_formal_are_separate_fresh_phases(tmp_path, monkeypatch):
    from scripts.run_v122_remote import run
    commands = []
    monkeypatch.setattr("scripts.run_v122_remote.subprocess.run",
                        lambda command, **kwargs: commands.append(command))
    run(tmp_path, tmp_path / "model", tmp_path / "source", tmp_path / "reference", tmp_path / "run")
    assert len(commands) == 3
    assert commands[0][2] == "scripts.preflight_v122"
    assert commands[1][2] == "scripts.train_v122" and "smoke" in commands[1]
    assert commands[2][2] == "scripts.train_v122" and "formal" in commands[2]
    assert commands[1][commands[1].index("--output-dir") + 1] != commands[2][commands[2].index("--output-dir") + 1]
