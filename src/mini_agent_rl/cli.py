"""中文 CLI：默认使用 FakeModel，确保无 GPU/网络也能演示。"""
from pathlib import Path
import asyncio, json
import typer
from .domain import Task
from .model import FakeModelClient
from .tools import SearchTool, ToolRegistry
from .reward import *
from .runtime import AgentRunner
from .storage import SQLiteStore

app=typer.Typer(help="Mini Agent RL：离线、RL-ready 的最小 Agent 框架")
def make_runner(db: str, corpus: str):
    store=SQLiteStore(db); registry=ToolRegistry(); registry.register(SearchTool(json.loads(Path(corpus).read_text(encoding="utf-8")))); reward=CompositeReward([ExactMatchReward(), EvidenceReward(), SearchCostPenalty(), RepeatedSearchPenalty(), InvalidActionPenalty()]); return AgentRunner(FakeModelClient(), registry, reward, store), store
@app.command()
def init(db: str="mini_agent.db"): SQLiteStore(db).close(); typer.echo(f"已初始化 SQLite: {db}")
@app.command()
def run(task: str="examples/tasks.jsonl", group_size: int=4, db: str="mini_agent.db", corpus: str="examples/corpus.json"):
    runner, store=make_runner(db, corpus); tasks=[Task(**json.loads(line)) for line in Path(task).read_text(encoding="utf-8").splitlines() if line.strip()]
    async def go():
        for item in tasks:
            group=await runner.run_group(item, group_size); typer.echo(f"任务 {item.id}: group={group.id}, 成功率={group.success_rate:.0%}, 平均奖励={group.mean_reward:.3f}")
            for rid in group.rollout_ids:
                r=store.get_rollout(rid); typer.echo(f"  {rid}: {r.status.value}, reward={r.reward:.3f}, answer={r.final_answer}")
    asyncio.run(go()); store.close()
@app.command()
def show(rollout_id: str, db: str="mini_agent.db"): r=SQLiteStore(db).get_rollout(rollout_id); typer.echo(r.model_dump_json(indent=2) if r else "未找到 rollout")
@app.command(name="list")
def list_rollouts(status: str|None=None, db: str="mini_agent.db"): 
    for r in SQLiteStore(db).list_rollouts(status): typer.echo(f"{r.id} {r.status.value} reward={r.reward:.3f}")
@app.command()
def export(output: str="trajectories.jsonl", db: str="mini_agent.db"): SQLiteStore(db).export_jsonl(output); typer.echo(f"已导出: {output}")
if __name__ == "__main__": app()
