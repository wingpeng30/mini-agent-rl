import json
import pytest
from mini_agent_rl.model.constrained_evidence import addressed_observation, action_grammar, audit_selection
from mini_agent_rl.model.local import LocalModelConfig, parse_agent_action, first_json_object_end
from mini_agent_rl.prompts import system_prompt, DIRECT_SYSTEM


def test_lossless_index_special_characters_duplicate_documents():
    body='A met B. A was born in C.\n"Quoted" {braces} 漢字\\end\n'
    doc=dict(id="d",title="Answer is not here",text=body,score=2.0)
    raw=json.dumps(dict(ok=True,result=[doc,doc],searched_queries=["A"]),ensure_ascii=False)
    text,spans=addressed_observation(raw,"m4")
    for rank in (1,2):
        subset=[s for s in spans if s["rank"]==rank]
        assert ''.join(s["quote"] for s in subset)==body
        for s in subset:
            assert body[s["body_char_span"][0]:s["body_char_span"][1]]==s["quote"]
            assert text[s["rendered_char_span"][0]:s["rendered_char_span"][1]]==s["quote"]
    assert len(set(s["address"] for s in spans))==len(spans)
    assert all("Answer is not here" not in s["quote"] for s in spans)


def test_evidence_address_is_not_entailment_and_no_action_override():
    _,spans=addressed_observation(json.dumps({"result":[{"id":"d","text":"A met B."}]}),"m2")
    value=dict(evidence=[spans[0]["address"]],action="answer",answer="B")
    raw=json.dumps(value)
    audit=audit_selection(raw,spans)
    assert audit["addresses_valid"] and not audit["affects_dispatch"]
    assert audit["semantic_support"]=="requires_source_grounded_adjudication"
    assert parse_agent_action(raw)==parse_agent_action('{"action":"answer","answer":"B"}')
    value["evidence"]=["hidden"]
    assert not audit_selection(json.dumps(value),spans)["addresses_valid"]
    assert audit_selection('{"evidence":[],"action":"search","query":"A father"}',[]) ["addresses_valid"]


def test_default_compatibility_and_probability_safeguard():
    cfg=LocalModelConfig()
    assert cfg.action_constraint=="none" and cfg.observation_presentation=="legacy-json"
    assert system_prompt(False)==DIRECT_SYSTEM
    with pytest.raises(ValueError):LocalModelConfig(action_constraint="evidence-action-v1")
    with pytest.raises(ValueError):LocalModelConfig(action_constraint="json-action-v1",recompute_old_logprobs=False,enable_thinking=True)
    cfg=LocalModelConfig(action_constraint="json-action-v1",recompute_old_logprobs=False)
    assert cfg.action_constraint=="json-action-v1"


def test_first_json_escaping_and_address_bounds():
    raw=json.dumps(dict(evidence=[],action="answer",answer='"braces { }"'))
    assert first_json_object_end(raw+'{"action":"answer","answer":"later"}')==len(raw)
    assert 'evidence ::=' not in action_grammar("json-action-v1",["x"])
    assert 'address ::=' not in action_grammar("evidence-action-v1",[])
    with pytest.raises(ValueError):action_grammar("unknown",[])
