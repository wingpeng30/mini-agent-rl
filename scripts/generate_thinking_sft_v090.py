"""Launch the bounded, locally credentialed thinking-SFT teacher generation."""
from mini_agent_rl.training.thinking_sft import generate_thinking_sft


if __name__ == "__main__":
    print(generate_thinking_sft(
        "data/hotpot-agent-v041/train.jsonl",
        "data/thinking-sft-v090",
        "D:/Tracefix/.env",
        task_limit=500,
        api_budget_cny=30.0,
    ), flush=True)
