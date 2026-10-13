"""CPU-only checks of real batch assembly and independent action termination."""
from types import SimpleNamespace
import hashlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mini_agent_rl.domain import Message
from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient, _BatchActionStoppingCriteria


class Tokenizer:
    eos_token_id = 256
    pad_token_id = 0
    all_special_ids = [256]

    def convert_tokens_to_ids(self, value):
        return 256 if value in {"<|im_end|>", "<|endoftext|>"} else None

    def decode(self, values, skip_special_tokens=False):
        return "".join("<|im_end|>" if token == 256 else "<pad>" if token == 0
                       else "} trailing" if token == 257 else chr(token) for token in values)

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))


class Input(dict):
    pass


class Model:
    device = "cpu"

    def __init__(self, outputs):
        self.outputs = outputs
        self.config = SimpleNamespace(use_cache=False)
        self.calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append(dict(ids=input_ids.clone(), masks=attention_mask.clone(), kwargs=kwargs,
                               rng=torch.rand(3).tolist()))
        result = input_ids.clone()
        done = torch.zeros(len(self.outputs), dtype=torch.bool)
        for step in range(kwargs["max_new_tokens"]):
            new = [kwargs["pad_token_id"] if done[row] else
                   self.outputs[row][step] if step < len(self.outputs[row]) else 256
                   for row in range(len(self.outputs))]
            result = torch.cat([result, torch.tensor(new).unsqueeze(1)], dim=1)
            done |= kwargs["stopping_criteria"](result, None)
            if done.all():
                break
        return result


def make_client(monkeypatch, outputs, *, max_new_tokens=128, completion="first-json-v1", score=True):
    client = TransformersModelClient(LocalModelConfig(model_id="unused", device="cpu", deterministic=False,
        max_new_tokens=max_new_tokens, action_completion=completion, recompute_old_logprobs=score, timeout=None))
    tokenizer = Tokenizer()
    client._processor = SimpleNamespace(tokenizer=tokenizer, decode=tokenizer.decode)
    client._model = Model(outputs)
    seen_scores = []

    def inputs(messages, phase):
        length = int(messages[0].content)
        ids = list(range(1, length + 1))
        value = Input(input_ids=torch.tensor([ids]), attention_mask=torch.ones((1, length), dtype=torch.long))
        value._context_usage = dict(row_label=str(length), actual_prompt_tokens=length,
            prompt_sha256=hashlib.sha256(bytes(str(ids), "utf-8")).hexdigest(), phase=phase)
        return value

    def score_sync(messages, actions, with_grad, phase, prompt_ids, prompt_mask):
        seen_scores.append((list(prompt_ids), list(prompt_mask), list(actions)))
        return torch.zeros(len(actions))

    monkeypatch.setattr(client, "_inputs", inputs)
    monkeypatch.setattr(client, "_score_sync", score_sync)
    return client, seen_scores


def requests(lengths, *, sample=False):
    return [dict(messages=[Message(role="user", content=str(length))], tools=[], sample=sample,
                 temperature=1.0 if sample else 0.0) for length in lengths]


def test_batch_uses_one_forward_left_pads_and_keeps_independent_token_context(monkeypatch):
    raw = ['{"action":"answer","answer":"x"}',
           '{"action":"search","query":"brace } { and \\\" quote"}']
    client, scores = make_client(monkeypatch, [list(map(ord, text)) for text in raw])
    result = client._generate_batch_sync(requests([2, 5]), 123, "policy-0")
    assert len(client._model.calls) == 1
    assert result[0].prompt_token_ids == [0, 0, 0, 1, 2]
    assert result[0].prompt_attention_mask == [0, 0, 0, 1, 1]
    assert result[1].prompt_token_ids == [1, 2, 3, 4, 5]
    assert result[0].action_token_ids == list(map(ord, raw[0]))
    assert result[1].action_token_ids == list(map(ord, raw[1]))
    assert result[0].content == "x" and result[1].tool_call.name == "search"
    assert [row.usage["context_usage"]["row_label"] for row in result] == ["2", "5"]
    assert result[0].usage["context_usage"] is not result[1].usage["context_usage"]
    assert result[0].usage["prompt_sha256"] == hashlib.sha256(bytes(str([0, 0, 0, 1, 2]), "utf-8")).hexdigest()
    assert result[0].usage["prompt_tokens"] == 2
    assert result[0].usage["padded_prompt_tokens"] == 5
    assert client._model.calls[0]["kwargs"]["position_ids"].tolist() == [[0, 0, 0, 0, 1], [0, 1, 2, 3, 4]]
    assert result[0].usage["prompt_position_ids"] == [0, 0, 0, 0, 1]
    assert scores[0] == (result[0].prompt_token_ids, result[0].prompt_attention_mask, result[0].action_token_ids)
    assert all(len(row.action_mask) == len(row.action_token_ids) == len(row.old_logprobs) for row in result)


def test_per_row_first_json_eos_and_length_are_distinct_and_no_followup_pad_is_scored(monkeypatch):
    complete = '{"action":"answer","answer":"x"}'
    client, scores = make_client(monkeypatch, [list(map(ord, complete)), list(map(ord, "invalid")) + [256],
                                               [ord("{")] * 64], max_new_tokens=64)
    result = client._generate_batch_sync(requests([1, 2, 3]), 4, "v1")
    assert [row.usage["generation_stop_reason"] for row in result] == ["first_json", "eos", "length"]
    assert result[1].action_token_ids == list(map(ord, "invalid")) + [256]
    assert result[1].parse_error and result[2].parse_error
    assert result[1].usage["generated_message_end"]
    assert result[2].usage["output_truncated"] and not result[0].usage["output_truncated"]
    assert all(0 not in row[2] for row in scores)


def test_token_containing_json_end_and_extra_text_is_preserved_but_only_json_parsed(monkeypatch):
    prefix = '{"action":"answer","answer":"x"'
    client, _ = make_client(monkeypatch, [list(map(ord, prefix)) + [257]])
    result = client._generate_batch_sync(requests([2]), 3, "v1")[0]
    assert result.action_token_ids[-1] == 257
    assert result.raw_text.endswith("} trailing")
    assert result.content == "x" and result.usage["first_json_tail_chars"] == len(" trailing")


def test_eos_mode_does_not_end_at_first_json(monkeypatch):
    text = '{"action":"answer","answer":"x"}'
    client, _ = make_client(monkeypatch, [list(map(ord, text)) + [256]], completion="eos")
    result = client._generate_batch_sync(requests([1]), 1, "v0")[0]
    assert result.action_token_ids[-1] == 256 and result.usage["generation_stop_reason"] == "eos"
    assert result.content == "x"


def test_sampling_parameters_and_batch_rng_are_explicit_and_repeatable(monkeypatch):
    output = list(map(ord, '{"action":"answer","answer":"x"}'))
    client, _ = make_client(monkeypatch, [output], score=False)
    first = client._generate_batch_sync(requests([1], sample=True), 50, "policy-3")[0]
    client._generate_batch_sync(requests([1], sample=True), 50, "policy-3")
    call = client._model.calls[0]
    assert call["rng"] == client._model.calls[1]["rng"]
    assert call["kwargs"]["do_sample"] and call["kwargs"]["temperature"] == 1.0
    assert call["kwargs"]["top_p"] == call["kwargs"]["repetition_penalty"] == 1.0
    assert call["kwargs"]["top_k"] == 0
    assert first.usage["batch_seed"] == 50 and first.usage["policy_version"] == "policy-3"
    assert first.old_logprobs == []


def test_batch_rejects_mixed_sampling_configuration_before_model_use(monkeypatch):
    client, _ = make_client(monkeypatch, [])
    with pytest.raises(ValueError, match="same sampling"):
        client._generate_batch_sync(requests([1]) + requests([2], sample=True), 1, "v0")
    assert not client._model.calls


@pytest.mark.parametrize("value", [[], [dict(messages=[], tools=[], sample=False, temperature=0)],
    [dict(messages=[Message(role="user", content="1")], sample=True, temperature=0)],
    [dict(messages=[Message(role="user", content="1")], sample=False, temperature=float("nan"))]])
def test_batch_rejects_invalid_requests(monkeypatch, value):
    client, _ = make_client(monkeypatch, [])
    with pytest.raises(ValueError):
        client._generate_batch_sync(value, 1, "v0")


def test_async_batch_interface(monkeypatch):
    import asyncio
    from mini_agent_rl.domain import ModelResponse
    client = TransformersModelClient(LocalModelConfig(device="cpu", timeout=None))
    monkeypatch.setattr(client, "_generate_batch_sync", lambda items, seed, version: [ModelResponse(content="x")])
    async def direct(function, *arguments):
        return function(*arguments)
    # Do not create a Windows Proactor loop/socket pair in the CPU sandbox.
    monkeypatch.setattr(asyncio, "to_thread", direct)
    coroutine = client.generate_batch(requests([2]), 2, "v1")
    with pytest.raises(StopIteration) as finished:
        coroutine.send(None)
    result = finished.value.value
    assert result[0].content == "x"


def test_batch_stopper_keeps_first_stop_length_after_other_rows_continue():
    tokenizer = Tokenizer()
    stopper = _BatchActionStoppingCriteria(tokenizer, 1, 2, True)
    done = stopper(torch.tensor([[1, ord("{")], [1, ord("{")]]), None)
    assert not done.any()
    done = stopper(torch.tensor([[1, ord("{"), ord("}")], [1, ord("{"), ord('"')]]), None)
    assert done.tolist() == [True, False] and stopper.stop_lengths == [2, None]
    stopper(torch.tensor([[1, ord("{"), ord("}"), 0], [1, ord("{"), ord('"'), 256]]), None)
    assert stopper.stop_lengths == [2, 3]


def test_padded_policy_replay_uses_generation_positions_without_changing_unpadded_forward():
    class Forward:
        device = "cpu"
        config = SimpleNamespace(use_cache=True)

        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, attention_mask, **kwargs):
            self.calls.append(dict(ids=input_ids, masks=attention_mask, **kwargs))
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 300)))

    client = TransformersModelClient(LocalModelConfig(device="cpu"))
    client._model = Forward()
    values = client._score_sync([], [123, 125], False, "old", [0, 0, 1, 2], [0, 0, 1, 1])
    assert values.shape == (2,) and torch.isfinite(values).all()
    assert client._model.calls[0]["position_ids"].tolist() == [[0, 0, 0, 1, 2, 3]]
    client._score_sync([], [123, 125], False, "old", [1, 2], [1, 1])
    assert "position_ids" not in client._model.calls[1]
