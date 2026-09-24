from mini_agent_rl.domain import ModelResponse, Task, Transition
from mini_agent_rl.reward.functions import AnswerTokenF1Reward, CompositeReward, ExactMatchReward


def answer_transition():
    return Transition(rollout_id="r", step=0, messages=[], response=ModelResponse(content="x"), terminated=True, termination_reason="answer")


def test_token_f1_reward_is_bounded_and_partial():
    reward = AnswerTokenF1Reward()
    task = Task(id="t", question="q", answer="New York City")
    value = reward.evaluate(task, [answer_transition()], "New York")["answer_token_f1"]
    assert 0 < value < .20
    assert reward.evaluate(task, [answer_transition()], "New York City")["answer_token_f1"] == .20


def test_yes_no_and_empty_answers_require_exact_or_zero():
    reward = AnswerTokenF1Reward()
    task = Task(id="t", question="q", answer="yes")
    assert reward.evaluate(task, [answer_transition()], "yes indeed")["answer_token_f1"] == 0.0
    assert reward.evaluate(task, [answer_transition()], "Yes")["answer_token_f1"] == .20
    assert reward.evaluate(task, [answer_transition()], "")["answer_token_f1"] == 0.0


def test_invalid_action_cannot_receive_f1_reward():
    invalid = Transition(rollout_id="r", step=0, messages=[], response=ModelResponse(), terminated=True, termination_reason="invalid_action")
    task = Task(id="t", question="q", answer="answer")
    assert AnswerTokenF1Reward().evaluate(task, [invalid], "answer")["answer_token_f1"] == 0.0


def test_diagnostics_never_change_reward_sum():
    reward = CompositeReward([ExactMatchReward()], version="v098-answer-v1")
    task = Task(id="t", question="q", answer="answer")
    transition = Transition(rollout_id="r", step=0, messages=[], response=ModelResponse(content="answer"), observation={"result": [{"text": "answer"}]}, terminated=True, termination_reason="answer")
    assert sum(reward.evaluate(task, [transition], "answer").values()) == 1.0
    assert reward.diagnostics(task, [transition], "answer")["answer_string_hit"] == 1.0
