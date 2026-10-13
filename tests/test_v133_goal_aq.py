import json
from mini_agent_rl.model.grounded_action_audit import audit_grounded_action, visible_documents, map_documents
from mini_agent_rl.model.local import first_json_object_end, parse_agent_action
from mini_agent_rl.prompts import system_prompt, DIRECT_SYSTEM


def message(body):
    return [{"role":"tool","content":json.dumps({"result":[{"id":"d","title":"Not evidence","text":body}]})}]


def output(quote, **changes):
    value={"links":[{"subject":"A","relation":"parent","object":"B","document_id":"d","quote":quote}],
           "missing_relation":"","action":"answer","answer":"B"}
    value.update(changes)
    return json.dumps(value)


def test_audit_does_not_change_action_or_claim_entailment():
    raw=output("A met B.")
    audit=audit_grounded_action(raw,visible_documents(message("A met B.")))
    assert audit["schema_valid"] and audit["exact_quotes_valid"]
    assert audit["semantic_support"]=="requires_source_grounded_adjudication"
    assert audit["affects_dispatch"] is False  # meeting cannot prove parenthood
    assert parse_agent_action(raw)==parse_agent_action('{"action":"answer","answer":"B"}')


def test_titles_and_assistant_answers_are_not_evidence():
    messages=message("A was born in C.")+[{"role":"assistant","content":"Not evidence"}]
    assert not audit_grounded_action(output("Not evidence"),visible_documents(messages))["exact_quotes_valid"]
    assert not audit_grounded_action(output("C",links=[]),[]) ["errors"]


def test_schema_and_first_json_boundary():
    raw=output('Nested { brace } and "quoted" text')
    assert first_json_object_end(raw+'{"action":"search","query":"later"}')==len(raw)
    assert not audit_grounded_action('{"action":"answer","answer":"B"}',[])["schema_valid"]
    assert not audit_grounded_action(output("x",links=[{}, {}, {}]),[])["schema_valid"]
    assert not audit_grounded_action(raw[:-1],[])["schema_valid"]
    assert first_json_object_end(raw[:-1]) is None
    assert system_prompt(False)==DIRECT_SYSTEM
    assert system_prompt(False,"direct-v090-grounded-action-v1")!=DIRECT_SYSTEM


class CharacterTokenizer:
    def decode(self,ids,**kwargs): return "".join(chr(x) for x in ids)
    def __call__(self,text,**kwargs): return {"input_ids":[ord(x) for x in text],"offset_mapping":[(i,i+1) for i in range(len(text))]}


def test_exact_quote_mapping_with_escapes_and_duplicate_documents():
    body='A was born in C. "quoted" { braces }'
    messages=message(body)*2
    rendered="\n".join(m["content"] for m in messages)
    tokenizer=CharacterTokenizer()
    docs,offsets=map_documents(tokenizer,[ord(c) for c in rendered],messages)
    audit=audit_grounded_action(output('"quoted"'),docs,offsets)
    assert audit["exact_quotes_valid"]
    assert len(audit["links"][0]["occurrences"])==2
    assert all(o["quote_input_token_span"][1]>o["quote_input_token_span"][0] for o in audit["links"][0]["occurrences"])
