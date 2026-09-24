"""中文 CLI：默认使用 FakeModel，确保无 GPU/网络也能演示。"""
from pathlib import Path
import asyncio, json, time, math
import typer
from .domain import Task
from .model import FakeModelClient, OpenAICompatibleClient, LocalModelConfig, TransformersModelClient, local_environment_report
from .tools import SearchTool, ToolRegistry
from .reward import *
from .runtime import AgentRunner
from .storage import SQLiteStore
from .evaluation import evaluate_tasks, hotpot_sft_to_tasks_and_corpus, load_hotpot_environment, load_tasks, rl_rollout_metrics, write_report, diagnose_store, file_sha256
from .training import (generate_dataset, download_hotpot_subset, prepare_final_test, audit_sft_directory, audit_final_test, audit_rl_signal, compare_logprobs,
                       stable_same_instance, GRPOConfig, MinimalGRPOTrainer, preflight, run_experiment, inspect_experiment, resolve_memory_limit_gb)
from .training.sft import train_sft
from .training.thinking_sft import generate_thinking_sft
from .stop_probe import audit_probe_budget, check_split_overlap, compare_probe_reports, evaluate as evaluate_stop_probe, load_frozen_cases

app=typer.Typer(help="Mini Agent RL：离线、RL-ready 的最小 Agent 框架")
experiment_app = typer.Typer(help="v0.8.3：冻结配置、监督执行并检查实验产物。")
app.add_typer(experiment_app, name="experiment")


def _hard_timeout_expired(started: float, timeout_minutes: int, *, now: float | None = None) -> bool:
    """Return whether a positive hard timeout elapsed; zero disables it."""
    if timeout_minutes <= 0:
        return False
    current = time.monotonic() if now is None else now
    return current - started > timeout_minutes * 60

def _training_code_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths = {
        "cli": root / "cli.py",
        "runner": root / "runtime" / "runner.py",
        "local_model": root / "model" / "local.py",
        "grpo": root / "training" / "grpo.py",
        "reward": root / "reward" / "functions.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}

def _retrieval_fingerprint(environment, retriever_version: str, observation_encoding: str) -> dict:
    full = environment.fingerprint(retriever_version, observation_encoding)
    keys = ("source_data_sha256", "corpus_sha256", "corpus_document_count", "corpus_limit", "retriever_version", "observation_encoding")
    return {key: full[key] for key in keys}

def _assert_policy_fingerprint(expected: dict, actual: dict) -> None:
    keys = ("base_config_sha256", "adapter_config_sha256", "adapter_weight_sha256", "tokenizer_sha256", "dtype")
    mismatched = [key for key in keys if expected.get(key) != actual.get(key)]
    if mismatched:
        raise typer.BadParameter("稳定性报告与当前策略指纹不匹配：" + ",".join(mismatched))

def default_reward(exact_weight: float = 1.0, repeated_penalty: float = -0.5, evidence_stop_weight: float = 0.0,
                   incomplete_evidence_answer_penalty: float = 0.0, empty_evidence_answer_penalty: float = 0.0,
                   reward_version: str = "v096", over_search_penalty: float = -0.2):
    if reward_version not in {"v096", "v098-answer-v1", "v102-oversearch-zero"}:
        raise typer.BadParameter("reward_version 仅支持 v096、v098-answer-v1 或 v102-oversearch-zero")
    answer = [EvidenceReward()] if reward_version == "v096" else [AnswerTokenF1Reward()]
    functions = [ExactMatchReward(exact_weight), *answer, SupportCoverageReward(), SearchCostPenalty(), RepeatedSearchPenalty(repeated_penalty), InvalidActionPenalty(), MaxStepsPenalty(), UngroundedAnswerPenalty(), OverSearchAfterEvidencePenalty(over_search_penalty)]
    if evidence_stop_weight: functions.append(EvidenceStopReward(evidence_stop_weight))
    if incomplete_evidence_answer_penalty: functions.append(IncompleteEvidenceAnswerPenalty(incomplete_evidence_answer_penalty))
    if empty_evidence_answer_penalty: functions.append(EmptyEvidenceAnswerPenalty(empty_evidence_answer_penalty))
    return CompositeReward(functions, version=reward_version)
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

@app.command(name="stop-probe")
def stop_probe(
    data: str = "data/hotpot-agent-v041/validation.jsonl",
    train_data: str = "data/hotpot-agent-v041/train.jsonl",
    limit: int = 30,
    corpus_limit: int = 100,
    max_input_tokens: int = 768,
    observation_encoding: str = "legacy",
    adapter_path: str = "checkpoints/qwen35-08b-sft-v052", model_path: str = r"D:\qwen_08b",
    output: str = "reports/v085-stop-probe",
    frozen_cases: str = "reports/v085-stop-768/cases.jsonl", memory_limit_gb: float | None = None,
    enable_thinking: bool = False, max_new_tokens: int = 128, generation_timeout: float = 600.0,
):
    """运行冻结历史的一步停止诊断；人工历史不计入模型策略行为。"""
    audit = check_split_overlap(train_data, data, 100)
    if not audit["passed"]:
        raise typer.BadParameter("训练/诊断分割存在 ID 或问题重叠，已停止，不替换题目")
    root = Path(output)
    cases = load_frozen_cases(frozen_cases, data)
    result = asyncio.run(evaluate_stop_probe(cases, root, adapter_path, max_input_tokens, observation_encoding, memory_limit_gb, model_path, enable_thinking, max_new_tokens, generation_timeout))
    (root / "split-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"停止探针状态={result['status']}，原因={result['reason']}，目录={root}")

@app.command(name="compare-stop-probe")
def compare_stop_probe(left: str, right: str, output: str = "reports/v085-stop-budget-compare.json"):
    """逐题比较两个已冻结停止探针的动作，不重跑模型。"""
    result = compare_probe_reports(Path(left), Path(right), Path(output))
    typer.echo(f"已比较 {result['case_count']} 个 case，动作差异={result['changed_action_count']}，报告={output}")

@app.command(name="audit-stop-probe")
def audit_stop_probe(
    data: str = "data/hotpot-agent-v041/validation.jsonl", limit: int = 30, corpus_limit: int = 100,
    model_path: str = r"D:\qwen_08b", max_input_tokens: int = 768,
    observation_encoding: str = "compact-v1", output: str = "reports/v086-budget-audit.json",
    frozen_cases: str = "reports/v085-stop-768/cases.jsonl",
):
    """只用本地 tokenizer 审计证据压缩与最终 prompt，不加载模型权重或生成。"""
    result = audit_probe_budget(load_frozen_cases(frozen_cases, data), Path(output), model_path, max_input_tokens, observation_encoding)
    typer.echo(f"预算审计完成：case={result['case_count']}，报告={output}")
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

@app.command(name="generate-thinking-sft")
def generate_thinking_sft_command(source_data: str = "data/hotpot-agent-v041/train.jsonl", output_dir: str = "data/thinking-sft-v090", env_file: str = r"D:\Tracefix\.env", model: str = "deepseek-flash", task_limit: int = 500, max_steps: int = 3, api_budget_cny: float = 30.0, pilot_only: bool = False):
    """生成经真实 BM25-v2 回放和教师审计的思考模式 SFT 数据。"""
    if "v072" in source_data.lower() or "final-test" in source_data.lower():
        raise typer.BadParameter("思考 SFT 禁止读取封存 final-test")
    manifest = generate_thinking_sft(source_data, output_dir, env_file, model, task_limit, max_steps, api_budget_cny, pilot_only)
    typer.echo(f"思考 SFT 数据已生成：train={manifest['counts']['train']} validation={manifest['counts']['validation']}，目录={output_dir}")
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
@app.command(name="audit-rl-signal")
def audit_rl_signal_command(db: str = typer.Option(..., help="RL rollout SQLite 路径"), output: str = "reports/v080-signal-audit.json"):
    """只读检查 reward、advantage 和轨迹完整性。"""
    report = audit_rl_signal(db); write_report(report, output)
    typer.echo(f"RL 信号审计：通过={report['passed']}，报告={output}")

@experiment_app.command("preflight")
def experiment_preflight(config: str = typer.Option(..., help="实验 TOML 配置路径")):
    """只读验证输入、封存数据和独立运行目录。"""
    report = preflight(config)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise typer.Exit(2)

@experiment_app.command("run")
def experiment_run(config: str = typer.Option(..., help="实验 TOML 配置路径")):
    """监督冻结命令；30 分钟硬超时，失败不自动重试。"""
    result = run_experiment(config)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "COMPLETED":
        raise typer.Exit(1)

@experiment_app.command("inspect")
def experiment_inspect(run_dir: str = typer.Argument(..., help="独立实验运行目录")):
    """读取 manifest 与最后一次监督状态，不修改产物。"""
    typer.echo(json.dumps(inspect_experiment(run_dir), ensure_ascii=False, indent=2))
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
    corpus_limit: int | None = None,
    retriever_version: str = "keyword-v1", max_input_tokens: int = 2048,
    max_new_tokens: int = 128, enable_thinking: bool = False, generation_timeout: float = 600.0,
    seed: int = 42, observation_encoding: str = "compact-v3", anti_repeat_prompt: bool = False,
    prompt_variant: str = "direct-v090",
):
    """在可重复使用的开发集评测单个冻结 adapter，不访问最终测试集。"""
    if Path(data).name == "final-test.jsonl":
        raise typer.BadParameter("最终集请使用 evaluate-final，并显式传入 --confirm-final。")
    environment = load_hotpot_environment(data, task_limit=limit, corpus_limit=corpus_limit or limit)
    tasks, corpus = environment.tasks, environment.corpus
    store = SQLiteStore(db)
    registry = ToolRegistry(); registry.register(SearchTool(corpus, retriever_version))
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path, max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking, timeout=generation_timeout, gradient_checkpointing=False, observation_encoding=observation_encoding))
    runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1, base_seed=seed,
                         anti_repeat_prompt=anti_repeat_prompt, environment_fingerprint=environment.fingerprint(retriever_version, observation_encoding), enable_thinking=enable_thinking,
                         prompt_variant=prompt_variant)
    report = asyncio.run(evaluate_tasks(runner, tasks, 1))
    report.update({"protocol": "development_reusable", "model_path": model_path, "adapter_path": adapter_path,
                   "data": data, "task_limit": limit, "corpus_limit": corpus_limit or limit,
                   "retriever_version": retriever_version, "environment_fingerprint": environment.fingerprint(retriever_version, observation_encoding), "protocol_fingerprint": runner.protocol_fingerprint(), "runtime_config": {"seed": seed, "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens, "enable_thinking": enable_thinking, "observation_encoding": observation_encoding, "anti_repeat_prompt": anti_repeat_prompt, "prompt_version": runner.prompt_version, "prompt_variant": prompt_variant}})
    write_report(report, output)
    typer.echo(f"开发评测完成：准确率={report['exact_match_rate']:.0%}，平均奖励={report['mean_reward']:.3f}，报告={output}")
    store.close()


@app.command(name="verify-environment")
def verify_environment(
    data: str = "data/hotpot-agent-v041/train.jsonl",
    task_limit_small: int = 5,
    task_limit_large: int = 10,
    corpus_limit: int = 100,
    retriever_version: str = "bm25-v2",
    output: str = "reports/v084-environment-parity.json",
):
    """只读核验不同 task_limit 是否共享同一个固定 corpus。"""
    if "final-test" in data.lower() or "v072" in data.lower():
        raise typer.BadParameter("环境核验禁止读取封存 final-test")
    small = load_hotpot_environment(data, task_limit=task_limit_small, corpus_limit=corpus_limit)
    large = load_hotpot_environment(data, task_limit=task_limit_large, corpus_limit=corpus_limit)
    report = {
        "version": "v0.8.4", "data": data, "source_data_sha256": small.source_data_sha256,
        "corpus_limit": corpus_limit, "retriever_version": retriever_version,
        "small": small.fingerprint(retriever_version), "large": large.fingerprint(retriever_version),
        "task_prefix_equal": small.task_ids == large.task_ids[:task_limit_small],
        "corpus_hash_equal": small.corpus_sha256 == large.corpus_sha256,
        "passed": small.corpus_sha256 == large.corpus_sha256 and small.task_ids == large.task_ids[:task_limit_small],
    }
    write_report(report, output)
    typer.echo(f"环境一致性：{'通过' if report['passed'] else '失败'}，corpus_sha256={small.corpus_sha256}，报告={output}")


@app.command(name="evaluate-retriever")
def evaluate_retriever(
    model_path: str = r"D:\qwen_08b",
    adapter_path: str = "checkpoints/qwen35-08b-sft-v052",
    data: str = "data/hotpot-agent-v041/validation.jsonl",
    limit: int = 30,
    corpus_limit: int = 100,
    output: str = "reports/v084-retriever-ab.json",
    db_dir: str = "runs-v084-retriever-ab",
):
    """在同一固定环境中对 keyword-v1 与 bm25-v2 做 greedy A/B。"""
    if "final-test" in data.lower() or "v072" in data.lower():
        raise typer.BadParameter("检索器 A/B 禁止读取封存 final-test")
    environment = load_hotpot_environment(data, task_limit=limit, corpus_limit=corpus_limit)
    root = Path(db_dir); root.mkdir(parents=True, exist_ok=True)
    results = {}
    for version in ("keyword-v1", "bm25-v2"):
        db = root / f"{version}.db"
        store = SQLiteStore(db)
        registry = ToolRegistry(); registry.register(SearchTool(environment.corpus, version))
        client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path, deterministic=True, max_input_tokens=768))
        runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1, temperature=0.0, environment_fingerprint=environment.fingerprint(version))
        report = asyncio.run(evaluate_tasks(runner, environment.tasks, 1))
        report.update({"retriever_version": version, "db": str(db), "environment_fingerprint": environment.fingerprint(version), "adapter_path": adapter_path})
        results[version] = report
        store.close()
        typer.echo(f"[{version}] EM={report['strict_exact_match_rate']:.1%}，F1={report['token_f1']:.3f}，截断={((report.get('context_usage') or {}).get('phases') or {}).get('rollout_sampling', {}).get('truncation_rate', 0):.1%}")
    write_report({"version": "v0.8.4", "protocol": "retriever-ab", "data": data, "limit": limit, "corpus_limit": corpus_limit, "results": results}, output)
    typer.echo(f"检索器 A/B 完成：报告={output}")


def _bootstrap_delta(left: list[float], right: list[float], seed: int = 42, repeats: int = 2000) -> dict[str, float]:
    """对齐 task 顺序后做简单 paired bootstrap，避免依赖 numpy。"""
    import random
    if not left or len(left) != len(right):
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    deltas = [b - a for a, b in zip(left, right)]
    rng = random.Random(seed); means = []
    for _ in range(repeats):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {"mean": sum(deltas) / len(deltas), "low": means[int(repeats * 0.025)], "high": means[int(repeats * 0.975) - 1], "n": len(deltas)}


@app.command(name="evaluate-v084-matrix")
def evaluate_v084_matrix(
    model_path: str = r"D:\qwen_08b",
    data: str = "data/hotpot-agent-v041/test.jsonl",
    limit: int = 100,
    corpus_limit: int = 100,
    output: str = "reports/v084-development-matrix.json",
    db_dir: str = "runs-v084-development-matrix",
):
    """固定 100 条开发集，对比 SFT、历史 seed44 与 v0.8.4 checkpoint。"""
    if "final-test" in data.lower() or "v072" in data.lower():
        raise typer.BadParameter("v0.8.4 开发集评测拒绝封存 final-test.jsonl")
    environment = load_hotpot_environment(data, task_limit=limit, corpus_limit=corpus_limit)
    candidates = [
        ("sft-v052", "checkpoints/qwen35-08b-sft-v052"),
        ("grpo-v071-seed44", "checkpoints/qwen35-08b-grpo-v071-seed44"),
        ("grpo-v084-formal", "checkpoints/qwen35-08b-grpo-v084-formal"),
    ]
    root = Path(db_dir); root.mkdir(parents=True, exist_ok=True); results = {}
    for name, adapter in candidates:
        db = root / f"{name}.db"
        store = SQLiteStore(db); registry = ToolRegistry(); registry.register(SearchTool(environment.corpus, "bm25-v2"))
        client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter, deterministic=True, max_input_tokens=768))
        runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1, temperature=0.0, environment_fingerprint=environment.fingerprint("bm25-v2"))
        report = asyncio.run(evaluate_tasks(runner, environment.tasks, 1))
        diagnostics_path = Path(output).with_name(f"{Path(output).stem}-{name}-diagnostics.json")
        diagnostics = diagnose_store(store, data, diagnostics_path)
        details = diagnostics["details"]
        report.update({"name": name, "adapter_path": adapter, "db": str(db), "environment_fingerprint": environment.fingerprint("bm25-v2"), "diagnostics": str(diagnostics_path), "support_coverage": sum(float(x.get("supporting_document_coverage") or 0.0) for x in details) / (len(details) or 1), "empty_search_rate": sum(bool(x.get("empty_search_result")) for x in details) / (len(details) or 1), "legacy_truncation_rate": sum(bool(x.get("prompt_truncated")) for x in details) / (len(details) or 1)})
        results[name] = report; store.close()
        typer.echo(f"[{name}] EM={report['strict_exact_match_rate']:.1%}，F1={report['token_f1']:.3f}，reward={report['mean_reward']:.3f}")
    sft_details = json.loads(Path(results["sft-v052"]["diagnostics"]).read_text(encoding="utf-8"))["details"]
    comparisons = {}
    for name in ("grpo-v071-seed44", "grpo-v084-formal"):
        details = json.loads(Path(results[name]["diagnostics"]).read_text(encoding="utf-8"))["details"]
        by_task = {row["task_id"]: row for row in details}; ordered = [by_task[row["task_id"]] for row in sft_details if row["task_id"] in by_task]
        comparisons[name] = {metric: _bootstrap_delta([float(a.get(metric, 0.0)) for a in sft_details if a["task_id"] in by_task], [float(b.get(metric, 0.0)) for b in ordered]) for metric in ("strict_exact_match", "normalized_exact_match", "token_f1", "answered", "repeated_action")}
    matrix = {"version": "v0.8.4", "protocol": "fixed-development-100-greedy-bm25-v2", "data": data, "limit": limit, "corpus_limit": corpus_limit, "environment_fingerprint": environment.fingerprint("bm25-v2"), "models": results, "paired_bootstrap_deltas_vs_sft": comparisons}
    write_report(matrix, output)
    typer.echo(f"v0.8.4 开发集矩阵完成：报告={output}")
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
def train_sft_command(model_path: str = r"D:\qwen_08b", train_data: str = "data/hotpot-agent-v041/train.jsonl", validation_data: str = "data/hotpot-agent-v041/validation.jsonl", output_dir: str = "checkpoints/qwen35-08b-sft", max_steps: int | None = None, epochs: int = 2, max_length: int = 2048, seed: int = 42, batch_size: int = 1, gradient_accumulation_steps: int = 8, gradient_checkpointing: bool = True, adapter_path: str | None = None, enable_thinking: bool = False, learning_rate: float = 1e-4, save_steps: int = 50, eval_steps: int = 50, warmup_ratio: float = 0.05):
    """训练 Qwen3.5-0.8B LoRA adapter；只监督 assistant 的 search/answer token。"""
    metrics = train_sft(model_path, train_data, validation_data, output_dir, max_steps, epochs, max_length, seed, batch_size, gradient_accumulation_steps, gradient_checkpointing, adapter_path, enable_thinking, learning_rate, save_steps, eval_steps, warmup_ratio)
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
def collect_rl(model_path: str = r"D:\qwen_08b", adapter_path: str = "checkpoints/qwen35-08b-sft-v052", data: str = "data/hotpot-agent-v041/train.jsonl", task_limit: int = 10, corpus_limit: int | None = None, retriever_version: str = "keyword-v1", group_size: int = 4, temperature: float = 0.7, seed: int = 42, db: str = "rollouts-v060.db", output: str = "data/rl-v060/trajectories.jsonl", report: str = "reports/rl-collection-v060.json", max_steps: int = 3, timeout_minutes: int = 30, load_in_4bit: bool = False, max_input_tokens: int = 2048, max_new_tokens: int = 128, observation_encoding: str = "legacy", enable_thinking: bool = False, generation_timeout: float = 600.0, anti_repeat_prompt: bool = False, reward_version: str = "v096", evidence_stop_weight: float = 0.0, incomplete_evidence_answer_penalty: float = 0.0, empty_evidence_answer_penalty: float = 0.0, over_search_penalty: float = -0.2):
    """采集本地 sampled rollout、action logprob 和 group advantage，不更新权重。"""
    if Path(db).exists(): raise typer.BadParameter(f"采样数据库已存在，拒绝混入历史轨迹：{db}")
    environment = load_hotpot_environment(data, task_limit=task_limit, corpus_limit=corpus_limit or task_limit)
    tasks, corpus = environment.tasks, environment.corpus
    store = SQLiteStore(db); registry = ToolRegistry(); registry.register(SearchTool(corpus, retriever_version))
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path, deterministic=True, load_in_4bit=load_in_4bit, max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens, observation_encoding=observation_encoding, enable_thinking=enable_thinking, timeout=generation_timeout))
    reward = default_reward(evidence_stop_weight=evidence_stop_weight, incomplete_evidence_answer_penalty=incomplete_evidence_answer_penalty, empty_evidence_answer_penalty=empty_evidence_answer_penalty, reward_version=reward_version, over_search_penalty=over_search_penalty)
    runner = AgentRunner(client, registry, reward, store, max_steps=max_steps, max_concurrency=1, temperature=temperature, anti_repeat_prompt=anti_repeat_prompt, base_seed=seed, environment_fingerprint=environment.fingerprint(retriever_version, observation_encoding), enable_thinking=enable_thinking)
    async def go():
        started = time.monotonic()
        rollout_ids = []
        for index, task in enumerate(tasks, 1):
            if _hard_timeout_expired(started, timeout_minutes):
                raise TimeoutError(f"超过 {timeout_minutes} 分钟硬超时，未自动重试")
            group = await runner.run_group(task, group_size)
            rollout_ids.extend(group.rollout_ids)
            reasons = [store.get_rollout(rid).termination_reason for rid in group.rollout_ids]
            typer.echo(f"[{index}/{len(tasks)}] group={group.id} mean={group.mean_reward:.3f} std={group.std_reward:.3f} reasons={reasons} 显存GB={client.gpu_memory_peak_gb()}")
        return rollout_ids
    rollout_ids = asyncio.run(go()); store.export_jsonl(output)
    report_path = report
    report = rl_rollout_metrics(store, rollout_ids); report.update({"version": "v0.9.8", "temperature": temperature, "group_size": group_size, "retriever_version": retriever_version, "reward_config": runner._reward_config(), "environment_fingerprint": environment.fingerprint(retriever_version, observation_encoding), "policy_fingerprint": client.policy_fingerprint(), "runtime_config": {"load_in_4bit": load_in_4bit, "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens, "observation_encoding": observation_encoding, "enable_thinking": enable_thinking}})
    write_report(report, report_path); store.close()
    typer.echo(f"RL 轨迹已导出：{output}，数据库={db}，报告={report_path}")

@app.command(name="validate-logprob")
def validate_logprob(model_path: str = r"D:\qwen_08b", adapter_path: str = "checkpoints/qwen35-08b-sft-v052", data: str = "data/hotpot-agent-v041/train.jsonl", repeats: int = 10, output: str = "reports/logprob-stability-v061.json", db: str = "logprob-validation-v061.db", load_in_4bit: bool = False, max_input_tokens: int = 2048, max_new_tokens: int = 128, observation_encoding: str = "legacy", enable_thinking: bool = False, corpus_limit: int = 100, retriever_version: str = "bm25-v2", anti_repeat_prompt: bool = False):
    """验证同实例严格稳定性与 BF16/FP32 跨加载数值阈值，不联网、不修改权重。"""
    if repeats < 2: raise typer.BadParameter("--repeats 至少为 2")
    environment = load_hotpot_environment(data, task_limit=1, corpus_limit=corpus_limit)
    tasks, corpus = environment.tasks, environment.corpus
    # 先固定一次真实离散动作；之后所有前向均使用完全相同 messages/action token。
    if Path(db).exists(): raise typer.BadParameter(f"稳定性数据库已存在，拒绝混入历史轨迹：{db}")
    store = SQLiteStore(db); registry=ToolRegistry(); registry.register(SearchTool(corpus, retriever_version))
    runtime = {"load_in_4bit": load_in_4bit, "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens, "observation_encoding": observation_encoding, "enable_thinking": enable_thinking}
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path, deterministic=True, dtype="bfloat16", **runtime))
    runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1, temperature=0.0, anti_repeat_prompt=anti_repeat_prompt, base_seed=42, environment_fingerprint=environment.fingerprint(retriever_version, observation_encoding), enable_thinking=enable_thinking)
    group = asyncio.run(runner.run_group(tasks[0], 1)); rollout=store.get_rollout(group.rollout_ids[0]); transition=next((t for t in store.get_transitions(rollout.id) if t.response.action_token_ids), None)
    if transition is None: raise RuntimeError("无法取得用于稳定性验证的 action token；请检查本地 adapter 的动作协议")
    prompt_ids = transition.response.prompt_token_ids
    prompt_mask = transition.response.prompt_attention_mask
    action_ids = transition.response.action_token_ids or []
    if not prompt_ids or prompt_mask is None or len(prompt_ids) != len(prompt_mask):
        raise RuntimeError("稳定性验证缺少完整 prompt token/mask，拒绝非精确重放")
    def score_exact(instance):
        return instance.score_action_tensor(transition.messages, action_ids, False, prompt_ids, prompt_mask).detach().cpu().tolist()
    same = [score_exact(client) for _ in range(repeats)]
    def independent(dtype: str):
        other=TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path, deterministic=True, dtype=dtype, logprob_mode="fp32" if dtype == "float32" else "bf16", **runtime))
        result=score_exact(other); return result, other.policy_fingerprint()
    bf_left, bf_fp_left = independent("bfloat16"); bf_right, bf_fp_right = independent("bfloat16")
    fp_left, fp_fp_left = independent("float32"); fp_right, fp_fp_right = independent("float32")
    bf16=compare_logprobs(bf_left, bf_right); fp32=compare_logprobs(fp_left, fp_right); same_report=stable_same_instance(same)
    bf_gate = bf16.get("valid", False) and bf16["absolute_delta"]["p95"] <= 1e-3 and .99 <= bf16["ratio"]["p95"] <= 1.01 and .90 <= bf16["ratio"]["min"] and bf16["ratio"]["max"] <= 1.10
    report={"version":"v0.9.4", "runtime_config": runtime, "experiment_config": {"corpus_limit": corpus_limit, "retriever_version": retriever_version, "retrieval_fingerprint": _retrieval_fingerprint(environment, retriever_version, observation_encoding)},
            "protocol_fingerprint": runner.protocol_fingerprint(), "code_fingerprint": _training_code_fingerprint(),
            "replay": {"exact_prompt_tokens": len(prompt_ids), "action_tokens": len(action_ids)}, "same_instance":same_report, "bf16_cross_load":bf16, "fp32_cross_load":fp32,
            "gates":{"same_instance_passed":same_report["passed"], "bf16_cross_load_passed":bf_gate,
                     "fp32_diagnostic_target_passed":fp32.get("valid",False) and fp32["absolute_delta"]["max"] <= 1e-5,
                     "grpo_allowed":same_report["passed"]}, "policy_fingerprints":{"sampling":client.policy_fingerprint(), "bf16_left":bf_fp_left, "bf16_right":bf_fp_right, "fp32_left":fp_fp_left, "fp32_right":fp_fp_right}}
    write_report(report, output); store.close()
    typer.echo(f"logprob 验证完成：同实例={'通过' if same_report['passed'] else '失败'}，BF16跨加载={'通过' if bf_gate else '未达标'}，FP32诊断={'通过' if report['gates']['fp32_diagnostic_target_passed'] else '未达标'}，报告={output}")

@app.command(name="train-grpo")
def train_grpo(model_path: str = r"D:\qwen_08b", adapter_path: str = "checkpoints/qwen35-08b-sft-v052", data: str = "data/hotpot-agent-v041/train.jsonl", task_limit: int = 5, corpus_limit: int | None = None, retriever_version: str = "keyword-v1", group_size: int = 4, temperature: float = 0.7, output_dir: str = "checkpoints/qwen35-08b-grpo-v070-smoke", seed: int = 42, max_steps: int = 3, stability_report: str = "reports/logprob-stability-v061.json", db: str | None = None, report: str | None = None, load_in_4bit: bool = False, max_input_tokens: int = 2048, observation_encoding: str = "legacy", exact_weight: float = 1.0, repeated_penalty: float = -0.5, evidence_stop_weight: float = 0.0, incomplete_evidence_answer_penalty: float = 0.0, empty_evidence_answer_penalty: float = 0.0, reward_version: str = "v096", over_search_penalty: float = -0.2, memory_limit_gb: float | None = None, gradient_checkpointing: bool = False, max_new_tokens: int = 128, generation_timeout: float = 600.0, enable_thinking: bool = False, learning_rate: float = 5e-6, clip_epsilon: float = .2, kl_beta: float = .02, policy_epochs: int = 2, max_grad_norm: float = 1.0, min_effective_groups: int | None = None, anti_repeat_prompt: bool = False):
    """同实例采样、重算 old/reference logprob 并更新 LoRA；小规模 5×4 冒烟默认配置。"""
    report_path = Path(stability_report)
    if not report_path.exists():
        raise typer.BadParameter(f"未找到稳定性报告 {stability_report}；请先运行 validate-logprob")
    stability = json.loads(report_path.read_text(encoding="utf-8"))
    if not stability.get("gates", {}).get("same_instance_passed"):
        raise typer.BadParameter("同一模型实例的 logprob 稳定性门槛未通过，拒绝启动 GRPO")
    environment = load_hotpot_environment(data, task_limit=task_limit, corpus_limit=corpus_limit or task_limit)
    tasks, corpus = environment.tasks, environment.corpus
    run_stem = Path(output_dir).name
    db_path = db or str(Path(output_dir).parent / f"{run_stem}.db")
    report_path_out = report or str(Path("reports") / f"{run_stem}.json")
    if Path(db_path).exists(): raise typer.BadParameter(f"训练数据库已存在，拒绝混入历史轨迹：{db_path}")
    store=SQLiteStore(db_path); registry=ToolRegistry(); registry.register(SearchTool(corpus, retriever_version))
    # 训练上下文独立收紧到 1024；推理默认仍可使用 2048，避免改变基线评测行为。
    client=TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path, deterministic=True, max_input_tokens=max_input_tokens, load_in_4bit=load_in_4bit, observation_encoding=observation_encoding, gradient_checkpointing=gradient_checkpointing, max_new_tokens=max_new_tokens, timeout=generation_timeout, enable_thinking=enable_thinking))
    expected_runtime = {"load_in_4bit": load_in_4bit, "max_input_tokens": max_input_tokens,
                        "max_new_tokens": client.config.max_new_tokens, "observation_encoding": observation_encoding, "enable_thinking": enable_thinking}
    if stability.get("runtime_config") != expected_runtime:
        raise typer.BadParameter("稳定性报告的量化、上下文或编码配置不匹配；请先用当前训练配置重新运行 validate-logprob")
    runner=AgentRunner(client, registry, default_reward(exact_weight, repeated_penalty, evidence_stop_weight, incomplete_evidence_answer_penalty, empty_evidence_answer_penalty, reward_version, over_search_penalty), store, max_steps=max_steps, max_concurrency=1, temperature=temperature, anti_repeat_prompt=anti_repeat_prompt, base_seed=seed, environment_fingerprint=environment.fingerprint(retriever_version, observation_encoding), enable_thinking=enable_thinking)
    experiment_fingerprint = {"corpus_limit": corpus_limit or task_limit, "retriever_version": retriever_version,
                              "retrieval_fingerprint": _retrieval_fingerprint(environment, retriever_version, observation_encoding)}
    if stability.get("experiment_config") != experiment_fingerprint:
        raise typer.BadParameter("稳定性报告的任务语料或检索器指纹不匹配")
    if stability.get("protocol_fingerprint") != runner.protocol_fingerprint():
        raise typer.BadParameter("稳定性报告的 prompt、历史策略或步数不匹配")
    if stability.get("code_fingerprint") != _training_code_fingerprint():
        raise typer.BadParameter("稳定性报告对应的训练代码已经变化；请重新运行 validate-logprob")
    # 加载后的指纹会在进入训练前再次核对；这里先捕获“采样起点”而非路径字符串。
    client._load(); expected=client.policy_fingerprint()
    _assert_policy_fingerprint(stability.get("policy_fingerprints", {}).get("sampling", {}), expected)
    memory_limit = resolve_memory_limit_gb(memory_limit_gb, local_environment_report().get("gpu_memory_gb"))
    trainer=MinimalGRPOTrainer(client, runner, store, GRPOConfig(learning_rate=learning_rate, clip_epsilon=clip_epsilon, kl_beta=kl_beta, policy_epochs=policy_epochs, max_grad_norm=max_grad_norm, max_reserved_memory_gb=memory_limit, min_effective_groups=min_effective_groups))
    result=asyncio.run(trainer.train(tasks, group_size, output_dir, expected))
    result["environment_fingerprint"] = environment.fingerprint(retriever_version, observation_encoding)
    result["reward_config"] = runner._reward_config()
    result["retriever_version"] = retriever_version
    result["temperature"] = temperature
    result["configured_memory_limit_gb"] = memory_limit
    write_report(result, report_path_out); store.close()
    typer.echo(f"GRPO 状态={result['status']}：groups={len(result['groups'])}，已变更LoRA参数={len(result['changed_lora_parameters'])}，adapter={output_dir}，数据库={db_path}，报告={report_path_out}")
    if result["status"] != "COMPLETED":
        raise typer.Exit(1)

@app.command(name="context-length-sweep")
def context_length_sweep(
    model_path: str = r"D:\qwen_08b",
    adapter_path: str = "checkpoints/qwen35-08b-sft-v052",
    data: str = "data/hotpot-agent-v041/validation.jsonl",
    task_limit: int = 10,
    lengths: str = "384,768,1024,2048",
    output: str = "reports/v082-context-length-sweep.json",
    db_dir: str = "runs-v082-context-length-sweep",
    memory_limit_gb: float | None = None,
):
    """同一冻结策略顺序执行上下文长度推理诊断；触发显存门槛后停止更长档位。"""
    if "final-test" in data.lower() or "v072" in data.lower():
        raise typer.BadParameter("上下文调参实验禁止读取封存 final-test")
    budgets = [int(item.strip()) for item in lengths.split(",") if item.strip()]
    if budgets != sorted(set(budgets)) or any(value < 128 for value in budgets):
        raise typer.BadParameter("lengths 必须是递增且不重复的 token 整数")
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(data, task_limit)
    root = Path(db_dir); root.mkdir(parents=True, exist_ok=True)
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path,
                                                       deterministic=True, max_input_tokens=budgets[0]))
    memory_limit_gb = resolve_memory_limit_gb(memory_limit_gb, local_environment_report().get("gpu_memory_gb"))
    client._load(); results = []
    for budget in budgets:
        db_path = root / f"context-{budget}.db"
        if db_path.exists():
            raise typer.BadParameter(f"实验数据库已存在，拒绝混入历史轨迹：{db_path}")
        client.config.max_input_tokens = budget
        client.reset_context_records(); client.reset_memory_stats()
        store = SQLiteStore(db_path); registry = ToolRegistry(); registry.register(SearchTool(corpus))
        runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1,
                             temperature=0.0, anti_repeat_prompt=True, base_seed=42)
        started = time.perf_counter()
        report = asyncio.run(evaluate_tasks(runner, tasks, 1))
        report.update({"context_budget": budget, "elapsed_seconds": time.perf_counter() - started,
                       "db": str(db_path), "data": data, "task_limit": task_limit})
        memory = report.get("gpu_memory", {})
        observed = max(float(memory.get(key) or 0.0) for key in ("peak_allocated_gb", "peak_reserved_gb", "device_process_gb"))
        finite = all(math.isfinite(float(report.get(key, 0.0))) for key in ("mean_reward", "token_f1", "exact_match_rate"))
        report["gate"] = {"observed_memory_gb": observed, "memory_limit_gb": memory_limit_gb,
                          "finite": finite, "no_infrastructure_failure": report["failed_rollout_count"] == 0,
                          "passed": finite and report["failed_rollout_count"] == 0 and observed <= memory_limit_gb}
        results.append(report); store.close()
        write_report({"version": "v0.8.2", "status": "RUNNING", "results": results}, output)
        typer.echo(f"context={budget} EM={report['strict_exact_match_rate']:.1%} F1={report['token_f1']:.3f} "
                   f"repeat={report['repeated_search_rate']:.1%} memory={observed:.3f}GB gate={report['gate']['passed']}")
        if not report["gate"]["passed"]:
            break
    final = {"version": "v0.8.2", "experiment_id": "mini-agent-rl-v082-context-budget",
             "status": "COMPLETED" if len(results) == len(budgets) else "STOPPED_BY_GATE",
             "model_path": model_path, "adapter_path": adapter_path, "data": data,
             "requested_lengths": budgets, "results": results}
    write_report(final, output)
    typer.echo(f"上下文推理诊断完成：状态={final['status']}，报告={output}")

@app.command(name="probe-context-backward")
def probe_context_backward(
    model_path: str = r"D:\qwen_08b",
    adapter_path: str = "checkpoints/qwen35-08b-sft-v052",
    data: str = "data/hotpot-agent-v041/validation.jsonl",
    max_input_tokens: int = 2048,
    output: str = "reports/v082-backward-probe.json",
    db: str = "runs-v082-backward-probe.db",
    memory_limit_gb: float | None = None,
    observation_encoding: str = "legacy",
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = False,
    max_new_tokens: int = 512,
    enable_thinking: bool = False,
    generation_timeout: float = 600.0,
):
    """执行一次可配置精度的 teacher-forced backward；不创建 optimizer，也不更新权重。"""
    if "final-test" in data.lower() or "v072" in data.lower():
        raise typer.BadParameter("显存探测禁止读取封存 final-test")
    if Path(db).exists():
        raise typer.BadParameter(f"探测数据库已存在，拒绝覆盖：{db}")
    tasks, corpus = hotpot_sft_to_tasks_and_corpus(data, 1)
    store = SQLiteStore(db); registry = ToolRegistry(); registry.register(SearchTool(corpus))
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter_path,
                                                       deterministic=True, max_input_tokens=max_input_tokens,
                                                       load_in_4bit=load_in_4bit, observation_encoding=observation_encoding,
                                                       gradient_checkpointing=gradient_checkpointing,
                                                       max_new_tokens=max_new_tokens, enable_thinking=enable_thinking,
                                                       timeout=generation_timeout))
    memory_limit_gb = resolve_memory_limit_gb(memory_limit_gb, local_environment_report().get("gpu_memory_gb"))
    runner = AgentRunner(client, registry, default_reward(), store, max_steps=3, max_concurrency=1,
                         temperature=0.0, anti_repeat_prompt=True, base_seed=42)
    group = asyncio.run(runner.run_group(tasks[0], 1))
    rollout = store.get_rollout(group.rollout_ids[0])
    transitions = [item for item in store.get_transitions(rollout.id) if item.response.action_token_ids]
    if not transitions:
        raise RuntimeError("没有可用于 backward 探测的 assistant action token")
    transition = max(transitions, key=lambda item: int(item.response.usage.get("original_prompt_tokens", 0)))
    params = [parameter for _, parameter in client.prepare_for_training()]
    client.reset_context_records(); client.reset_memory_stats()
    started = time.perf_counter()
    values = client.score_action_tensor(transition.messages, transition.response.action_token_ids or [], True)
    loss = -values.mean(); loss.backward()
    grad_finite = all(parameter.grad is None or bool(__import__("torch").isfinite(parameter.grad).all()) for parameter in params)
    memory = client.gpu_memory_stats()
    observed = max(float(memory.get(key) or 0.0) for key in ("peak_allocated_gb", "peak_reserved_gb", "device_process_gb"))
    report = {"version": "v0.8.9", "context_budget": max_input_tokens, "load_in_4bit": load_in_4bit,
              "gradient_checkpointing": gradient_checkpointing, "max_new_tokens": max_new_tokens,
              "enable_thinking": enable_thinking, "generation_timeout": generation_timeout,
              "observation_encoding": observation_encoding,
              "updated_weights": False, "action_tokens": len(transition.response.action_token_ids or []),
              "loss": float(loss.detach()), "loss_finite": bool(__import__("torch").isfinite(loss)),
              "gradients_finite": grad_finite, "elapsed_seconds": time.perf_counter() - started,
              "gpu_memory": memory, "context_usage": client.context_usage_summary(),
              "gate": {"memory_limit_gb": memory_limit_gb, "observed_memory_gb": observed}}
    report["gate"]["passed"] = report["loss_finite"] and grad_finite and observed <= memory_limit_gb
    write_report(report, output); store.close()
    typer.echo(f"backward 探测：context={max_input_tokens} memory={observed:.3f}GB gate={report['gate']['passed']}，报告={output}")
@app.command(name="run-grpo-ablation")
def run_grpo_ablation(model_path: str = r"D:\qwen_08b", adapter_path: str = "checkpoints/qwen35-08b-sft-v052", data: str = "data/hotpot-agent-v041/train.jsonl", task_limit: int = 5, group_size: int = 4, output_dir: str = "checkpoints/v080-ablation", seed: int = 42):
    """登记 A/B/C 受控实验配置；拒绝封存 final-test 数据。"""
    if "final-test" in data.lower() or "v072" in data.lower():
        raise typer.BadParameter("v0.8.0 禁止使用封存 final-test 数据")
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    arms = {"A": {"exact_match": 1.0, "repeated_search": -0.5}, "B": {"exact_match": 1.0, "repeated_search": -0.75}, "C": {"exact_match": 1.5, "repeated_search": -0.5}}
    manifest = {"version": "v0.8.0", "model_path": model_path, "adapter_path": adapter_path, "data": data, "task_limit": task_limit, "group_size": group_size, "seed": seed, "arms": arms, "status": "CONFIGURED"}
    (root / "ablation-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"v0.8.0 三臂实验配置已写入：{root / 'ablation-manifest.json'}；请先通过 audit-rl-signal 后再启动训练")

if __name__ == "__main__": app()
