"""Small CPU correctness checks for the new driver; never loads a real model."""
import asyncio
import json
import threading
import time

import run_final_online_grpo as run
from mini_agent_rl.domain import ModelResponse, ToolCall


class Engine:
    documents = []

    def __init__(self):
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def search(self, query, mode):
        with self.lock:
            self.active += 1
            self.peak = max(self.active, self.peak)
        time.sleep(.05)
        with self.lock:
            self.active -= 1
        return [dict(id=query, title=query, text="Evidence for " + query)]


class Client:
    def __init__(self):
        self.waves = []

    async def generate_batch(self, requests, *, batch_seed, policy_version):
        self.waves.append(len(requests))
        result = []
        for request in requests:
            messages = request["messages"]
            question = messages[1].content
            tokens = dict(prompt_token_ids=[0, 1, 2], prompt_attention_mask=[0, 1, 1],
                          action_token_ids=[3, 4], usage=dict(context_usage=dict(row=question)))
            if len(messages) == 2:
                # All four search calls must be executed concurrently.
                result.append(ModelResponse(raw_text=json.dumps(dict(action="search", text=question)),
                                            parse_error="query field missing", **tokens))
            elif question == "repeat":
                result.append(ModelResponse(tool_calls=[ToolCall(name="search", arguments=dict(query=question))], **tokens))
            elif question == "invalid":
                result.append(ModelResponse(raw_text="broken", parse_error="bad JSON", **tokens))
            else:
                result.append(ModelResponse(content=question + " answer", raw_text="{}", **tokens))
        return result


def main():
    client, engine = Client(), Engine()
    cases = [dict(id=s, question=s) for s in ("a", "b", "repeat", "invalid")]
    values = asyncio.run(run.wavefront(client, engine, cases, sample=True, seed=42, policy_version="fixture"))
    assert client.waves == [4, 4]
    assert engine.peak == 4, "tool searches were not concurrent"
    assert [v["reason"] for v in values] == ["answer", "answer", "repeated_action", "invalid_action"]
    assert all(v["searches"] == 1 and v["model_calls"] == 2 for v in values)
    assert [v["trace"][0]["query"] for v in values] == [c["question"] for c in cases]
    assert all(v["trace"][0]["usage"]["protocol_recovery"] == "search_text_alias_v1" for v in values)
    for value in values:
        run.check_action(value["trace"][0])
    import torch
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    parameter.grad = torch.tensor([2.])
    run.finite_optimizer_step(optimizer, [parameter])
    assert torch.allclose(parameter, torch.tensor([.8]))
    parameter.grad = torch.tensor([float("nan")])
    try:
        run.finite_optimizer_step(optimizer, [parameter])
    except RuntimeError:
        pass
    else:
        raise AssertionError("nonfinite gradient accepted")
    assert run.metrics("Saint Petersburg", dict(answer="St Petersburg", answer_aliases=["Saint Petersburg"]))["normalized_exact_match"] == 1
    print("CPU driver checks passed: wavefront/tool concurrency, independent histories, termination, alias, padding, optimizer finite checks, existing alias reward")


if __name__ == "__main__":
    main()
