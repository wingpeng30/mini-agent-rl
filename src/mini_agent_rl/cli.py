"""中文 CLI：默认使用 FakeModel，确保无 GPU/网络也能演示。"""
from pathlib import Path
import asyncio, json, time
import typer
from .domain import Task
from .model import FakeModelClient, OpenAICompatibleClient, LocalModelConfig, TransformersModelClient, local_environment_report
from .tools import SearchTool, ToolRegistry
from .reward import *
from .runtime import AgentRunner
from .storage import SQLiteStore
from .evaluation import evaluate_tasks, hotpot_sft_to_tasks_and_corpus, load_tasks, rl_rollout_metrics, write_report, diagnose_store, file_sha256
from .training import (generate_dataset, download_hotpot_subset, prepare_final_test, audit_sft_directory, audit_final_test, compare_logprobs,
                       stable_same_instance, GRPOConfig, MinimalGRPOTrainer)
from .training.sft import train_sft

app=typer.Typer(help="Mini Agent RL：离线、RL-ready 的最小 Agent 框架")
def default_reward():
    return CompositeReward([ExactMatchReward(), EvidenceReward(), SupportCoverageReward(), SearchCostPenalty(), RepeatedSearchPenalty(), InvalidActionPenalty(), MaxStepsPenalty(), UngroundedAnswerPenalty()])
def make_runner(db: str, corpus: str, backend: str = "fake", base_url: str = "https://api.deepseek.com", model: str | None = None, max_concurrency: int = 4, load_in_4bit: bool = False, max_steps: int = 5, seed: int = 0):
    store=SQLiteStore(db); registry=ToolRegistry(); registry.register(SearchTool(json.loads(Path(corpus).read_text(encoding="utf-8"))))
    reward=default_reward()
    if backend == "deepseek":
        if not __import__("os").environ.get("DEEPSEEK_API_KEY"):
            raise typer.BadParameter("请先设置 DEEPSEEK_API_KEY；密钥不会写入本地文件。")
        client = OpenAICompatibleClient(base_url=base_url, model=model or "deepseek-chat")
    elif backend == "local":
        client = TransformersModelClient(LocalModelConfig(model_id=model or r"D:\qwen_08b", load_in_4bit=load_in_4bit))
    else:
        client = FakeModelClient()
    return AgentRunner(client, registry, reward, store, max_steps=max_steps, max_concurrency=max_concurrency, base_seed=seed), store
@app.command()
def init(db: str="mini_agent.db"): SQLiteStore(db).close(); typer.echo(f"已初始化 SQLite: {db}")
@app.command()
def run(task: str="examples/tasks.jsonl", group_size: int=4, db: str="mini_agent.db", corpus: str="examples/corpus.json", backend: str="fake", base_url: str="https://api.deepseek.com", model: str|None=None, max_concurrency: int=4, load_in_4bit: bool=False, max_steps: int=3):
    """运行任务。--backend deepseek 时只从 DEEPSEEK_API_KEY 读取密钥。"""
    runner, store=make_runner(db, corpus, backend, base_url, model, max_concurrency, load_in_4bit, max_steps); tasks=load_tasks(task)
    async def go():
        for item in tasks:
            group=await runner.run_group(item, group_size); typer.echo(f"任务 {item.id}: group={group.id}, 成功率={group.success_rate:.0%}, 平均奖励={group.mean_reward:.3f}")
            for rid in group.rollout_ids:
                r=store.get_rollout(rid); typer.echo(f"  {rid}: {r.status.value}, reward={r.reward:.3f}, answer={r.final_answer}")
    asyncio.run(go()); store.close()
@app.command()
def evaluate(task: str="examples/tasks.jsonl", group_size: int=4, db: str="benchmark.db", corpus: str="examples/corpus.json", output: str="reports/baseline.json", backend: str="fake", base_url: str="https://api.deepseek.com", model: str|None=None, max_concurrency: int=4, load_in_4bit: bool=False, max_steps: int=3):
    """批量评测并保存准确率、奖励、搜索成本等基线指标。"""
    runner, store=make_runner(db, corpus, backend, base_url, model, max_concurrency, load_in_4bit, max_steps)
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
@app.command(name="local-check")
def local_check():
    """检查本地模型依赖和显卡；不会加载或下载模型。"""
    typer.echo(json.dumps(local_environment_report(), ensure_ascii=False, indent=2))
@app.command(name="local-smoke")
def local_smoke(model_path: str = r"D:\qwen_08b", db: str = "local-smoke.db", corpus: str = "examples/corpus.json", max_new_tokens: int = 128):
    """使用本地模型完成一条 search -> answer 冒烟轨迹。"""
    runner, store = make_runner(db, corpus, backend="local", model=model_path, max_concurrency=1, max_steps=3)
    task = Task(id="local-smoke", question="故宫位于哪个城市？", answer="故宫是中国明清两代的皇家宫殿，位于北京。")
    group = asyncio.run(runner.run_group(task, 1))
    result = store.get_rollout(group.rollout_ids[0])
    client = runner.model
    load_seconds = getattr(client, "load_seconds", None)
    peak_memory = client.gpu_memory_peak_gb() if hasattr(client, "gpu_memory_peak_gb") else None
    typer.echo(f"本地模型冒烟：状态={result.status.value}，奖励={result.reward:.3f}，答案={result.final_answer}，加载秒数={load_seconds}，峰值显存GB={peak_memory}，数据库={db}")
    store.close()
@app.command(name="generate-sft")
def generate_sft(corpus: str = "examples/corpus.json", output_dir: str = "data/sft-v040", train: int = 800, validation: int = 100, test: int = 100, seed: int = 42):
    """离线生成并校验 SFT search→observation→answer 数据，不调用模型 API。"""
    manifest = generate_dataset(corpus, output_dir, train, validation, test, seed)
    typer.echo(f"SFT 数据已生成：{manifest['counts']}，目录={output_dir}，校验错误={len(manifest['errors'])}")
@app.command(name="download-hotpot")
def download_hotpot(output_dir: str = "data/hotpot-agent-v041", train: int = 600, validation: int = 100, test: int = 100, max_scanned: int = 50000):
    """流式导入 HotpotQA 多跳检索数据，并进行 supporting-doc 级别切分。"""
    manifest = download_hotpot_subset(output_dir, train, validation, test, max_scanned)
    typer.echo(f"HotpotQA 导入完成：{manifest['counts']}，文档数={manifest['document_counts']}，扫描={manifest['scanned']}，目录={output_dir}")
@app.command(name="prepare-final-test")
def prepare_final_test_command(source_dir: str = "data/hotpot-agent-v041", output_dir: str = "data/hotpot-agent-v072", count: int = 300, max_scanned: int = 50000):
    """创建与已有 train/validation/development 文档隔离的 HotpotQA 最终测试集。"""
    manifest = prepare_final_test(source_dir, output_dir, count, max_scanned)
    typer.echo(f"最终测试集已创建：{manifest['final_test']['count']} 条，扫描={manifest['final_test']['scanned']}，SHA256={manifest['final_test']['sha256']}，目录={output_dir}")
@app.command(name="audit-sft")
def audit_sft(data_dir: str = "data/hotpot-agent-v041", output: str = "reports/sft-audit.json"):
    """审计 SFT split、问题重复、搜索分布和 supporting-document 泄漏。"""
    report = audit_sft_directory(data_dir)
    write_report(report, output)
    typer.echo(f"SFT 审计完成：文档隔离={report['passed_document_isolation']}，报告={output}")
@app.command(name="audit-final-test")
def audit_final_test_command(source_dir: str = "data/hotpot-agent-v041", final_dir: str = "data/hotpot-agent-v072", output: str = "reports/final-test-audit.json"):
    """审计最终测试集的 task、supporting-document 与 manifest 隔离。"""
    report = audit_final_test(source_dir, final_dir)
    write_report(report, output)
    typer.echo(f"最终测试集审计：通过={report['passed']}，task重叠={report['task_id_overlap']}，文档重叠={report['supporting_document_overlap']}，报告={output}")
@app.command(name="evaluate-final")
def evaluate_final(
    model_path: str = r"D:\qwen_08b",
    adapter_path: str = "checkpoints/qwen35-08b-sft-v052",
    data: str = "data/hotpot-agent-v072/final-test.jsonl",
    limit: int = 300,
    output: str = "reports/final-evaluation.json",
    db: str = "final-evaluation.db",
    confirm_final: bool = typer.Option(False, "--confirm-final", help="确认模型已冻结；最终测试仅允许执行一次。"),
):
    """对冻结 checkpoint 执行一次最终评测，避免把最终集误用于调参。"""
    if not confirm_final:
        raise typer.BadParameter("最终测试集不可用于调参。确认 checkpoint 后，请追加 --confirm-final。")
    final_root = Path(data).parent
    audit = audit_final_test("data/hotpot-agent-v041", final_root)
    if not audit["passed"]:
        raise typer.BadParameter(f"最终测试集隔离审计未通过：{audit}")
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(data, limit)
    store = SQLiteStore(db)
    registry = ToolRegistry(); registry.register(SearchTool(corpus))
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path))
    runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1)
    report = asyncio.run(evaluate_tasks(runner, tasks, 1))
    report.update({"protocol": "final_test_once", "model_path": model_path, "adapter_path": adapter_path,
                   "data": data, "data_sha256": audit["final_sha256"], "task_limit": limit})
    write_report(report, output)
    typer.echo(f"最终评测完成：准确率={report['exact_match_rate']:.0%}，平均奖励={report['mean_reward']:.3f}，报告={output}")
    store.close()
@app.command(name="evaluate-adapter")
def evaluate_adapter(
    model_path: str = r"D:\qwen_08b",
    adapter_path: str = "checkpoints/qwen35-08b-grpo-v071-seed43",
    data: str = "data/hotpot-agent-v041/test.jsonl",
    limit: int = 30,
    output: str = "reports/development-evaluation.json",
    db: str = "development-evaluation.db",
):
    """在可重复使用的开发集评测单个冻结 adapter，不访问最终测试集。"""
    if Path(data).name == "final-test.jsonl":
        raise typer.BadParameter("最终集请使用 evaluate-final，并显式传入 --confirm-final。")
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(data, limit)
    store = SQLiteStore(db)
    registry = ToolRegistry(); registry.register(SearchTool(corpus))
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path))
    runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1)
    report = asyncio.run(evaluate_tasks(runner, tasks, 1))
    report.update({"protocol": "development_reusable", "model_path": model_path, "adapter_path": adapter_path,
                   "data": data, "task_limit": limit})
    write_report(report, output)
    typer.echo(f"开发评测完成：准确率={report['exact_match_rate']:.0%}，平均奖励={report['mean_reward']:.3f}，报告={output}")
    store.close()
@app.command(name="analyze-rollouts")
def analyze_rollouts(db: str = typer.Option(..., "--db", help="SQLite rollout 数据库"), data: str = "data/hotpot-agent-v041/test.jsonl", output: str = "reports/rollout-diagnostics.json"):
    """只读分析已落库 rollout 的答案、检索、截断和失败类型。"""
    if Path(data).name == "final-test.jsonl":
        raise typer.BadParameter("v0.7.3 诊断不读取已封存的 final-test.jsonl")
    store = SQLiteStore(db)
    report = diagnose_store(store, data, output)
    store.close()
    typer.echo(f"轨迹诊断完成：rollout={report['rollout_count']}，EM={report['overall']['strict_exact_match_rate']:.1%}，F1={report['overall']['token_f1']:.3f}，报告={output}")

@app.command(name="evaluate-matrix")
def evaluate_matrix(
    model_path: str = r"D:\qwen_08b",
    data: str = "data/hotpot-agent-v041/test.jsonl",
    limit: int = 100,
    output: str = "reports/v073-matrix.json",
    db_dir: str = "runs-v073",
):
    """在固定开发集上顺序比较 SFT 与三个 v0.7.1 GRPO adapter。"""
    if Path(data).name == "final-test.jsonl":
        raise typer.BadParameter("矩阵评测拒绝 final-test.jsonl；最终测试集已封存")
    candidates = [
        ("sft-v052", "checkpoints/qwen35-08b-sft-v052"),
        ("grpo-v071-seed42", "checkpoints/qwen35-08b-grpo-v071-audit-smoke"),
        ("grpo-v071-seed43", "checkpoints/qwen35-08b-grpo-v071-seed43"),
        ("grpo-v071-seed44", "checkpoints/qwen35-08b-grpo-v071-seed44"),
    ]
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(data, limit)
    root = Path(db_dir); root.mkdir(parents=True, exist_ok=True)
    results = []
    for name, adapter in candidates:
        db = root / f"{name}.db"; report_path = Path(output).with_name(f"{Path(output).stem}-{name}.json")
        store = SQLiteStore(db); registry = ToolRegistry(); registry.register(SearchTool(corpus))
        client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter, deterministic=True))
        runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1, temperature=0.0)
        report = asyncio.run(evaluate_tasks(runner, tasks, 1))
        diagnostics = diagnose_store(store, data, report_path.with_name(f"{report_path.stem}-diagnostics.json"))
        report.update({"name": name, "adapter_path": adapter, "data_sha256": file_sha256(data), "db": str(db), "diagnostics": str(report_path.with_name(f"{report_path.stem}-diagnostics.json")), "policy_fingerprint": client.policy_fingerprint()})
        write_report(report, report_path); results.append(report); store.close()
        typer.echo(f"[{name}] EM={report['strict_exact_match_rate']:.1%}，F1={report['token_f1']:.3f}，reward={report['mean_reward']:.3f}")
    matrix = {"version": "v0.7.3", "protocol": "fixed-development-100-greedy", "data": data, "data_sha256": file_sha256(data), "task_limit": limit, "models": results}
    write_report(matrix, output)
    typer.echo(f"矩阵评测完成：{len(results)} 个 checkpoint，报告={output}")
@app.command(name="train-sft")
def train_sft_command(model_path: str = r"D:\qwen_08b", train_data: str = "data/hotpot-agent-v041/train.jsonl", validation_data: str = "data/hotpot-agent-v041/validation.jsonl", output_dir: str = "checkpoints/qwen35-08b-sft", max_steps: int | None = None, epochs: int = 2, max_length: int = 1024, seed: int = 42):
    """训练 Qwen3.5-0.8B LoRA adapter；只监督 assistant 的 search/answer token。"""
    metrics = train_sft(model_path, train_data, validation_data, output_dir, max_steps, epochs, max_length, seed)
    typer.echo(f"SFT 训练完成：训练样本={metrics['train_samples']}，验证样本={metrics['validation_samples']}，峰值显存GB={metrics['peak_gpu_memory_gb']}，输出={output_dir}")
@app.command(name="compare-adapter")
def compare_adapter(model_path: str = r"D:\qwen_08b", adapter_path: str = "checkpoints/qwen35-08b-sft-smoke", data: str = "data/hotpot-agent-v041/test.jsonl", limit: int = 8, output: str = "reports/adapter-comparison.json", max_steps: int = 3, db_prefix: str | None = None):
    """在相同的本地 Hotpot 环境中比较基础模型与 LoRA adapter。"""
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(data, limit)
    reward = default_reward()
    async def run_variant(name: str, adapter: str | None):
        store = SQLiteStore(f"{db_prefix or name}-comparison.db")
        registry = ToolRegistry(); registry.register(SearchTool(corpus))
        client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter))
        runner = AgentRunner(client, registry, reward, store, max_steps=max_steps, max_concurrency=1)
        report = await evaluate_tasks(runner, tasks, 1)
        store.close()
        return report
    adapter_version = "v0.7.0" if "grpo" in adapter_path.lower() else "v0.5.2"
    report = {"version": adapter_version, "data": data, "task_limit": limit, "adapter_path": adapter_path,
              "base": asyncio.run(run_variant("base", None)), "adapter": asyncio.run(run_variant("adapter", adapter_path))}
    write_report(report, output)
    typer.echo(f"对比完成：base 准确率={report['base']['exact_match_rate']:.0%}，adapter 准确率={report['adapter']['exact_match_rate']:.0%}，报告={output}")
@app.command(name="collect-rl")
def collect_rl(model_path: str = r"D:\qwen_08b", adapter_path: str = "checkpoints/qwen35-08b-sft-v052", data: str = "data/hotpot-agent-v041/train.jsonl", task_limit: int = 10, group_size: int = 4, temperature: float = 0.7, seed: int = 42, db: str = "rollouts-v060.db", output: str = "data/rl-v060/trajectories.jsonl", max_steps: int = 3, timeout_minutes: int = 30):
    """采集本地 sampled rollout、action logprob 和 group advantage，不更新权重。"""
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(data, task_limit)
    store = SQLiteStore(db); registry = ToolRegistry(); registry.register(SearchTool(corpus))
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path))
    runner = AgentRunner(client, registry, default_reward(), store, max_steps=max_steps, max_concurrency=1, temperature=temperature, anti_repeat_prompt=True, base_seed=seed)
    async def go():
        started = time.monotonic()
        rollout_ids = []
        for index, task in enumerate(tasks, 1):
            if time.monotonic() - started > timeout_minutes * 60:
                raise TimeoutError(f"超过 {timeout_minutes} 分钟硬超时，未自动重试")
            group = await runner.run_group(task, group_size)
            rollout_ids.extend(group.rollout_ids)
            reasons = [store.get_rollout(rid).termination_reason for rid in group.rollout_ids]
            typer.echo(f"[{index}/{len(tasks)}] group={group.id} mean={group.mean_reward:.3f} std={group.std_reward:.3f} reasons={reasons} 显存GB={client.gpu_memory_peak_gb()}")
        return rollout_ids
    rollout_ids = asyncio.run(go()); store.export_jsonl(output)
    report = rl_rollout_metrics(store, rollout_ids); report.update({"version": "v0.6.0", "temperature": temperature, "group_size": group_size})
    report_path = "reports/rl-collection-v060.json"; write_report(report, report_path); store.close()
    typer.echo(f"RL 轨迹已导出：{output}，数据库={db}，报告={report_path}")

@app.command(name="validate-logprob")
def validate_logprob(model_path: str = r"D:\qwen_08b", adapter_path: str = "checkpoints/qwen35-08b-sft-v052", data: str = "data/hotpot-agent-v041/train.jsonl", repeats: int = 10, output: str = "reports/logprob-stability-v061.json"):
    """验证同实例严格稳定性与 BF16/FP32 跨加载数值阈值，不联网、不修改权重。"""
    if repeats < 2: raise typer.BadParameter("--repeats 至少为 2")
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(data, 1)
    # 先固定一次真实离散动作；之后所有前向均使用完全相同 messages/action token。
    store = SQLiteStore("logprob-validation-v061.db"); registry=ToolRegistry(); registry.register(SearchTool(corpus))
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path, deterministic=True, dtype="bfloat16"))
    runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1, temperature=0.0, base_seed=42)
    group = asyncio.run(runner.run_group(tasks[0], 1)); rollout=store.get_rollout(group.rollout_ids[0]); transition=next((t for t in store.get_transitions(rollout.id) if t.response.action_token_ids), None)
    if transition is None: raise RuntimeError("无法取得用于稳定性验证的 action token；请检查本地 adapter 的动作协议")
    async def same_scores(): return [await client.score_action(transition.messages, transition.response.action_token_ids or []) for _ in range(repeats)]
    same = asyncio.run(same_scores())
    def independent(dtype: str):
        other=TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path, deterministic=True, dtype=dtype, logprob_mode="fp32" if dtype == "float32" else "bf16"))
        result=asyncio.run(other.score_action(transition.messages, transition.response.action_token_ids or [])); return result, other.policy_fingerprint()
    bf_left, bf_fp_left = independent("bfloat16"); bf_right, bf_fp_right = independent("bfloat16")
    fp_left, fp_fp_left = independent("float32"); fp_right, fp_fp_right = independent("float32")
    bf16=compare_logprobs(bf_left, bf_right); fp32=compare_logprobs(fp_left, fp_right); same_report=stable_same_instance(same)
    bf_gate = bf16.get("valid", False) and bf16["absolute_delta"]["p95"] <= 1e-3 and .99 <= bf16["ratio"]["p95"] <= 1.01 and .90 <= bf16["ratio"]["min"] and bf16["ratio"]["max"] <= 1.10
    report={"version":"v0.6.1", "same_instance":same_report, "bf16_cross_load":bf16, "fp32_cross_load":fp32,
            "gates":{"same_instance_passed":same_report["passed"], "bf16_cross_load_passed":bf_gate,
                     "fp32_diagnostic_target_passed":fp32.get("valid",False) and fp32["absolute_delta"]["max"] <= 1e-5,
                     "grpo_allowed":same_report["passed"]}, "policy_fingerprints":{"sampling":client.policy_fingerprint(), "bf16_left":bf_fp_left, "bf16_right":bf_fp_right, "fp32_left":fp_fp_left, "fp32_right":fp_fp_right}}
    write_report(report, output); store.close()
    typer.echo(f"logprob 验证完成：同实例={'通过' if same_report['passed'] else '失败'}，BF16跨加载={'通过' if bf_gate else '未达标'}，FP32诊断={'通过' if report['gates']['fp32_diagnostic_target_passed'] else '未达标'}，报告={output}")

@app.command(name="train-grpo")
def train_grpo(model_path: str = r"D:\qwen_08b", adapter_path: str = "checkpoints/qwen35-08b-sft-v052", data: str = "data/hotpot-agent-v041/train.jsonl", task_limit: int = 5, group_size: int = 4, output_dir: str = "checkpoints/qwen35-08b-grpo-v070-smoke", seed: int = 42, max_steps: int = 3, stability_report: str = "reports/logprob-stability-v061.json", db: str | None = None, report: str | None = None):
    """同实例采样、重算 old/reference logprob 并更新 LoRA；小规模 5×4 冒烟默认配置。"""
    report_path = Path(stability_report)
    if not report_path.exists():
        raise typer.BadParameter(f"未找到稳定性报告 {stability_report}；请先运行 validate-logprob")
    stability = json.loads(report_path.read_text(encoding="utf-8"))
    if not stability.get("gates", {}).get("same_instance_passed"):
        raise typer.BadParameter("同一模型实例的 logprob 稳定性门槛未通过，拒绝启动 GRPO")
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(data, task_limit)
    run_stem = Path(output_dir).name
    db_path = db or str(Path(output_dir).parent / f"{run_stem}.db")
    report_path_out = report or str(Path("reports") / f"{run_stem}.json")
    store=SQLiteStore(db_path); registry=ToolRegistry(); registry.register(SearchTool(corpus))
    # 训练上下文独立收紧到 1024；推理默认仍可使用 2048，避免改变基线评测行为。
    client=TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path, deterministic=True, max_input_tokens=1024))
    runner=AgentRunner(client, registry, default_reward(), store, max_steps=max_steps, max_concurrency=1, temperature=.7, anti_repeat_prompt=True, base_seed=seed)
    # 加载后的指纹会在进入训练前再次核对；这里先捕获“采样起点”而非路径字符串。
    client._load(); expected=client.policy_fingerprint()
    trainer=MinimalGRPOTrainer(client, runner, store, GRPOConfig())
    result=asyncio.run(trainer.train(tasks, group_size, output_dir, expected))
    write_report(result, report_path_out); store.close()
    typer.echo(f"GRPO 训练完成：groups={len(result['groups'])}，已变更LoRA参数={len(result['changed_lora_parameters'])}，adapter={output_dir}，数据库={db_path}，报告={report_path_out}")
if __name__ == "__main__": app()
