from mini_agent_rl.training.sft import AssistantOnlyCollator, encode_assistant_only


class FakeProcessor:
    """测试用稳定模板：每条消息写入 role token 与 content token。"""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        values = []
        for message in messages:
            values.extend([1 if message["role"] == "assistant" else 2, *[ord(char) for char in message["content"]]])
        return values

    class _Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(char) for char in text]}

    tokenizer = _Tokenizer()


def test_assistant_only_labels_exclude_tool_tokens():
    messages = [
        {"role": "system", "content": "s"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": "A"}, {"role": "tool", "content": "OBS"},
        {"role": "assistant", "content": "B"},
    ]
    encoded = encode_assistant_only(FakeProcessor(), messages)
    assert 65 in encoded["labels"] and 66 in encoded["labels"]
    assert ord("O") not in encoded["labels"]
    assert encoded["labels"].count(-100) > 0


def test_collator_pads_and_preserves_ignore_labels():
    batch = AssistantOnlyCollator(0)([
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [3], "attention_mask": [1], "labels": [3]},
    ])
    assert batch["input_ids"].shape == (2, 2)
    assert batch["labels"][1, 1].item() == -100
