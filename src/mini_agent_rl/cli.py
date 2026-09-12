"""中文 CLI：默认使用 FakeModel，确保无 GPU/网络也能演示。"""
from pathlib import Path
import asyncio, json
import typer
from .domain import Task
from .model import FakeModelClient, OpenAICompatibleClient
from .tools import SearchTool, ToolRegistry
from .reward import *
from .runtime import AgentRunner
from .storage import SQLiteStore
from .evaluation import evaluate_tasks, load_tasks, write_report

app=typer.Typer(help="Mini Agent RL：离线、RL-ready 的最小 Agent 框架")
def make_runner(db: str, corpus: str, backend: str = "fake", base_url: str = "https://api.deepseek.com", model: str = "deepseek-v4-flash", max_concurrency: int = 4):
    store=SQLiteStore(db); registry=ToolRegistry(); registry.register(SearchTool(json.loads(Path(corpus).read_text(encoding="utf-8"))))
    reward=CompositeReward([ExactMatchReward(), EvidenceReward(), SearchCostPenalty(), RepeatedSearchPenalty(), InvalidActionPenalty()])
    if backend == "deepseek":
        if not __import__("os").environ.get("DEEPSEEK_API_KEY"):
            raise typer.BadParameter("请先设置 DEEPSEEK_API_KEY；密钥不会写入本地文件。")
        client = OpenAICompatibleClient(base_url=base_url, model=model)
    else:
        client = FakeModelClient()
    return AgentRunner(client, registry, reward, store, max_concurrency=max_concurrency), store
@app.command()
def init(db: str="mini_agent.db"): SQLiteStore(db).close(); typer.echo(f"已初始化 SQLite: {db}")
@app.command()
def run(task: str="examples/tasks.jsonl", group_size: int=4, db: str="mini_agent.db", corpus: str="examples/corpus.json", backend: str="fake", base_url: str="https://api.deepseek.com", model: str="deepseek-v4-flash", max_concurrency: int=4):
    """运行任务。--backend deepseek 时只从 DEEPSEEK_API_KEY 读取密钥。"""
    runner, store=make_runner(db, corpus, backend, base_url, model, max_concurrency); tasks=load_tasks(task)
    async def go():
        for item in tasks:
            group=await runner.run_group(item, group_size); typer.echo(f"任务 {item.id}: group={group.id}, 成功率={group.success_rate:.0%}, 平均奖励={group.mean_reward:.3f}")
            for rid in group.rollout_ids:
                r=store.get_rollout(rid); typer.echo(f"  {rid}: {r.status.value}, reward={r.reward:.3f}, answer={r.final_answer}")
    asyncio.run(go()); store.close()
@app.command()
def evaluate(task: str="examples/tasks.jsonl", group_size: int=4, db: str="benchmark.db", corpus: str="examples/corpus.json", output: str="reports/baseline.json", backend: str="fake", base_url: str="https://api.deepseek.com", model: str="deepseek-v4-flash", max_concurrency: int=4):
    """批量评测并保存准确率、奖励、搜索成本等基线指标。"""
    runner, store=make_runner(db, corpus, backend, base_url, model, max_concurrency)
    report=asyncio.run(evaluate_tasks(runner, load_tasks(task), group_size)); report.update({"backend": backend, "model": model})
    write_report(report, output); typer.echo(f"评测完成：准确率={report['exact_match_rate']:.0%}，平均奖励={report['mean_reward']:.3f}，报告={output}")
    store.close()
@app.command()
def show(rollout_id: str, db: str="mini_agent.db"): r=SQLiteStore(db).get_rollout(rollout_id); typer.echo(r.model_dump_json(indent=2) if r else "未找到 rollout")
@app.command(name="list")
def list_rollouts(status: str|None=None, db: str="mini_agent.db"): 
    for r in SQLiteStore(db).list_rollouts(status): typer.echo(f"{r.id} {r.status.value} reward={r.reward:.3f}")
@app.command()
def export(output: str="trajectories.jsonl", db: str="mini_agent.db"): SQLiteStore(db).export_jsonl(output); typer.echo(f"已导出: {output}")
if __name__ == "__main__": app()
