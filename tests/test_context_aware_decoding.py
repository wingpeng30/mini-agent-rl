import json
import pytest
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context_aware_decoding import reference_messages, search_count
from mini_agent_rl.model.local import LocalModelConfig, first_json_object_end, parse_agent_action, ActionParseError


def test_reference_removes_only_document_content_and_preserves_duplicates():
    doc = dict(id="id", title="Title", text='Alice says "{escaped}"; Bob is her father.', score=3.0, other=[1])
    value = dict(ok=True, result=[doc, dict(doc)], searched_queries=["Alice father"])
    messages = [Message(role="user", content="question"), Message(role="tool", content=json.dumps(value), tool_call_id="call")]
    reference = reference_messages(messages)
    actual = json.loads(reference[1].content)
    expected = json.loads(json.dumps(value))
    for d in expected["result"]: d.update(title="", text="")
    assert actual == expected and len(actual["result"]) == 2
    assert json.loads(messages[1].content) == value
    assert reference[0] == messages[0] and reference[1].tool_call_id == "call"
    assert search_count(messages) == 1


def test_default_and_invalid_modes():
    assert LocalModelConfig().context_aware_decoding is None
    assert LocalModelConfig(context_aware_decoding=dict(alpha=0), recompute_old_logprobs=False).context_aware_decoding.alpha == 0
    for options in (dict(enable_thinking=True), dict(action_constraint="json-action-v1"), dict(recompute_old_logprobs=True)):
        with pytest.raises(ValueError):
            LocalModelConfig(context_aware_decoding={}, **{"recompute_old_logprobs":False, **options})


def test_first_object_not_skipped_or_repaired():
    raw = '{"action":"search","query":"A \\"nested\\" {entity}"}{"action":"answer","answer":"hidden"}'
    end = first_json_object_end(raw)
    assert end and raw[end:].startswith('{"action":"answer"')
    _, tool = parse_agent_action(raw[:end])
    assert tool.name == "search"
    bad = '{"action":"wrong"}{"action":"answer","answer":"hidden"}'
    with pytest.raises(ActionParseError): parse_agent_action(bad[:first_json_object_end(bad)])


def test_cached_reference_follows_same_prefix_and_fp32_math():
    import torch
    from types import SimpleNamespace
    from mini_agent_rl.model.context_aware_decoding import ContextAwareLogitsProcessor
    class Cache:
        def __init__(self): self.length = 0
        def get_seq_length(self): return self.length
    class FakeModel:
        def prepare_inputs_for_generation(self, ids, next_sequence_length, **kw):
            return dict(input_ids=ids if next_sequence_length is None else ids[:,-1:], **kw)
        def __call__(self, input_ids, past_key_values, **kw):
            seen.append(input_ids.tolist());past_key_values.length += input_ids.shape[1]
            return SimpleNamespace(logits=torch.tensor([[[1., 3., 0.]]]), past_key_values=past_key_values)
        def _update_model_kwargs_for_generation(self, out, kw): return kw
    seen=[]
    p=object.__new__(ContextAwareLogitsProcessor)
    p.model=FakeModel();p.primary_length=2;p.alpha=1.;p.ids=torch.tensor([[9,8,7]])
    p.kwargs=dict(past_key_values=Cache());p.steps=[];p.reference_tokens=[9,8,7]
    score=torch.tensor([[3.,2.,1.]],dtype=torch.bfloat16)
    assert p(torch.tensor([[4,5]]),score).tolist()==[[5.,1.,2.]]
    p(torch.tensor([[4,5,0]]),score)
    assert seen==[[[9,8,7]],[[0]]] and p.kwargs['past_key_values'].length==4
    assert p.audit()['reference_forward_passes']==2
