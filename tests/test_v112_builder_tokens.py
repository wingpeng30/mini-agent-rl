import pytest

from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import budget_messages, ContextBudgetError
from scripts.build_v112 import normalize_prompt_tokens, render_prompt


@pytest.mark.parametrize('value', [[0, 1, 2], [[0, 1, 2]], {'input_ids': [0, 1, 2]}, {'input_ids': [[0, 1, 2]]}])
def test_supported_single_prompt_shapes(value):
    assert normalize_prompt_tokens(value) == [0, 1, 2]


@pytest.mark.parametrize('value', [None, [], [[]], [[1], [2]], [[[1]]], [1, [2]],
                                    [True], [1.0], ['1'], [-1], (1, 2), {}, {'input_ids': None}])
def test_invalid_shapes_and_token_types(value):
    with pytest.raises(ValueError, match='prompt_token'):
        normalize_prompt_tokens(value)


class BatchedProcessor:
    def __init__(self, length):
        self.length = length

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': True, 'add_generation_prompt': True, 'enable_thinking': False}
        assert messages[-1] == {'role': 'user', 'content': 'question'}
        return [[1] * self.length]


def test_render_counts_tokens_not_batch_dimension():
    prompt = render_prompt(BatchedProcessor(200), [Message(role='user', content='question')])
    assert len(prompt) == 200


def test_single_batch_cannot_bypass_input_budget():
    messages = [Message(role='system', content='system'), Message(role='user', content='question')]
    processor = BatchedProcessor(2049)
    with pytest.raises(ContextBudgetError):
        budget_messages(messages, 2048, lambda ms: len(render_prompt(processor, ms)),
                        lambda text: len(text), observation_encoding='compact-v3')
