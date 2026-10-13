"""Goal AF: product-identity-safe retrieval evaluation; never trains weights."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import statistics
import sys
import time
import shutil
from pathlib import Path

import run_v133_goal_ae_corpus_repair as ae
import run_v133_goal_aa_audit_confirmation as aa
import run_v133_goal_p_system as p
import run_v133_goal_q_search as q
import run_v133_goal_y_three_class_transfer as y
import run_v133_goal_z_multihop_sft as z

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-af-retrieval-validation"
PUBLISHED_ADAPTER = ROOT / "runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final"
PUBLISHED_ADAPTER_SHA256 = "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52"
BASE_MODEL = Path("/root/autodl-tmp/qwen_08b")
REMOTE_ADAPTER = Path("/root/autodl-tmp/mini-agent-rl/runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final")
ARM_RETRIEVERS = {"A": "bm25-v2", "B": "bm25-v3-paragraph", "C": "bm25-v4-page-paragraph"}
SPLITS = ("goal_z_dev", "old_regression", "goal_y_regression", "aa_regression")
COUNTS = {"goal_z_dev": 64, "old_regression": 32, "goal_y_regression": 48, "aa_regression": 64}
DIAGNOSTICS = {"first_retrieval_failed", "bridge_entity_not_used", "answered_before_sufficient_evidence",
               "continued_after_sufficient_evidence", "sufficient_evidence_wrong_answer", "actual_input_truncation"}
MAX_TRAJECTORIES = 624
MAX_MODEL_CALLS = 2496
MAX_SEARCHES = 1872
CONFIRM_MANIFEST = ROOT / "runs/v133-goal-ab-evidence-decision/fresh-confirm-manifest.json"


def release_identity_matches(release: dict, actual_hash: str) -> bool:
    path = str(release.get("adapter", "")).replace("\\", "/")
    return (release.get("published") is True and release.get("selected_route") == "trajectory"
            and release.get("selected_seed") == 42
            and path.endswith("runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final")
            and actual_hash == PUBLISHED_ADAPTER_SHA256)


def approval_matches(approved_hash: str, manifest_hash: str) -> bool:
    return approved_hash == manifest_hash


def base_model_path_matches(adapter_base_path: str, expected_base_path: Path) -> bool:
    # Adapter configs may be written on Windows and evaluated on Linux.
    normalize = lambda value: str(value).replace("\\", "/").rstrip("/").split("/")[-1]
    return normalize(adapter_base_path) == normalize(expected_base_path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def files() -> dict[str, Path]:
    values = {
        "adapter": PUBLISHED_ADAPTER / "adapter_model.safetensors",
        "adapter_config": PUBLISHED_ADAPTER / "adapter_config.json",
        "published_decision": aa.OUT / "product-decision.json",
        "tokenizer": q.p.BASE / "tokenizer.json",
        "tokenizer_config": q.p.BASE / "tokenizer_config.json",
        "base_config": q.p.BASE / "config.json",
        "q_runner": Path(q.__file__), "ae_runner": Path(ae.__file__), "aa_runner": Path(aa.__file__),
        "z_runner": Path(z.__file__), "search_runner": ROOT / "src/mini_agent_rl/tools/search.py",
        "model_runner": ROOT / "src/mini_agent_rl/model/local.py",
        "af_runner": Path(__file__), "goal_z_rubric": z.OUT / "evidence-rubric.json",
        "goal_z_dev_cases": z.OUT / "selected-dev.jsonl", "goal_aa_confirm_cases": z.OUT / "selected-confirm.jsonl",
        "old_cases": p.CASES, "goal_y_cases": y.OUT / "dev-cases.jsonl",
        "goal_aa_review_results": aa.OUT / "confirmation-revised-results.jsonl",
        "goal_aa_decision": aa.OUT / "product-decision.json",
        "ae_static_manifest": ae.STATIC / "manifest.json",
        "fresh_confirmation_manifest": CONFIRM_MANIFEST,
    }
    for split, path in ae.original_paths().items():
        values[f"original_corpus_{split}"] = path
    for split, path in ae.paragraph_paths().items():
        values[f"paragraph_corpus_{split}"] = path
    values.update({f"base_weight_{i}": path for i, path in enumerate(sorted(q.p.BASE.glob("*.safetensors")))})
    for pattern in ("tokenizer*.json", "special_tokens_map.json", "vocab.json", "merges.txt",
                    "processor*.json", "preprocessor_config.json", "generation_config.json"):
        for path in q.p.BASE.glob(pattern):
            values[f"base_artifact_{path.name}"] = path
    historical = {
        "goal_z_dev": z.OUT / "eval-dev-42-trajectory.jsonl",
        "old_regression": z.OUT / "eval-old-42-trajectory.jsonl",
        "goal_y_regression": z.OUT / "eval-y-42-trajectory.jsonl",
        "aa_regression": aa.OUT / "eval-confirm-42-trajectory.jsonl",
    }
    values.update({f"historical_{key}": value for key, value in historical.items()})
    return values


def software_facts() -> dict:
    import torch
    import transformers
    import peft
    import tokenizers
    import accelerate
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        driver = None
    return {
        "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__, "tokenizers": tokenizers.__version__, "accelerate": accelerate.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi_gpu_driver_memory": driver,
        "dtype": "bfloat16", "quantization": False, "eval_mode": True,
        "deterministic_algorithms": "warn_only per Goal Q; CUDA may not be bitwise deterministic",
        "environment": {key: os.environ.get(key) for key in
                        ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")},
    }


def validate_release_identity() -> dict:
    release = read(aa.OUT / "product-decision.json")
    adapter = files()["adapter"]
    actual = sha(adapter)
    if not release_identity_matches(release, actual):
        raise RuntimeError("AA release record does not identify the published Goal Z trajectory seed 42")
    base_weights = sorted(q.p.BASE.glob("*.safetensors"))
    if not base_weights:
        raise RuntimeError("base model weight files are missing")
    base_hashes = {path.name: sha(path) for path in base_weights}
    if len(base_hashes) != 1 or next(iter(base_hashes.values())) != \
       "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696":
        raise RuntimeError(f"base model weight identity differs from frozen Goal AF precondition: {base_hashes}")
    if not CONFIRM_MANIFEST.is_file() or read(CONFIRM_MANIFEST).get("confirmation_accessed") is not False:
        raise RuntimeError("fresh confirmation manifest missing or already opened")
    static = read(ae.STATIC / "manifest.json")
    if static.get("fresh_goal_ab_confirmation_accessed") is not False:
        raise RuntimeError("static snapshot says the confirmation was accessed")
    return {"adapter_sha256": actual, "release_adapter": release["adapter"], "base_weight_sha256": base_hashes,
            "release_status": release.get("status"), "fresh_confirmation_accessed": False,
            "fresh_confirmation_case_and_corpus_sha256": {
                "cases": read(CONFIRM_MANIFEST)["case_sha256"], "corpus": read(CONFIRM_MANIFEST)["corpus_sha256"]}}


def cohort_cases() -> dict[str, list[dict]]:
    cases = ae.frozen_cases()
    old_subsplits = {row["case_id"]: row["split"] for row in q.dev_cases()}
    cases["old_regression"] = [{**row, "subsplit": old_subsplits[row["case_id"]]}
                                for row in cases["old_regression"]]
    counts = {key: len(value) for key, value in cases.items()}
    if counts != COUNTS:
        raise RuntimeError(f"frozen case counts differ: {counts}")
    return cases


def replay_saved_queries() -> dict:
    """CPU-only retrieval replay over the exact queries saved in published Goal Z traces."""
    from transformers import AutoTokenizer
    from mini_agent_rl.tools import SearchTool

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True, trust_remote_code=False)
    total = 0
    rows_out = []
    for split, old_path in historical_paths().items():
        source_rows = ae.read_rows(old_path)
        b_tool = SearchTool(read(ae.paragraph_paths()[split]), "bm25-v3-paragraph")
        c_tool = SearchTool(read(ae.paragraph_paths()[split]), "bm25-v4-page-paragraph", tokenizer=tokenizer)
        for row in source_rows:
            for step_index, step in enumerate(row.get("trace", [])):
                query = step.get("query")
                if not query:
                    continue
                baseline = step.get("observation", {}).get("result", [])
                b_results = b_tool._bm25_index.search(query, top_k=3)
                c_results = c_tool._bm25_index.search(query, top_k=3)
                rows_out.append({"case_id": row["case_id"], "split": split, "step": step_index,
                    "query": query, "original_goal_z_results": baseline,
                    "bm25_v3_paragraph_results": b_results, "bm25_v4_page_paragraph_results": c_results,
                    "bm25_v4_provenance": c_tool._bm25_index.last_sources})
                total += 1
                if total % 50 == 0:
                    print(json.dumps({"cpu_replay_queries": total}, ensure_ascii=False), flush=True)
        del b_tool, c_tool
    if total != 391:
        raise RuntimeError(f"frozen Goal Z query count changed: expected 391, found {total}")
    replay_path = OUT / "cpu-retrieval-replay.jsonl"
    with replay_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows_out:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"queries": total, "sha256": sha(replay_path), "path": str(replay_path)}


def historical_paths() -> dict[str, Path]:
    return {"goal_z_dev": z.OUT / "eval-dev-42-trajectory.jsonl",
            "old_regression": z.OUT / "eval-old-42-trajectory.jsonl",
            "goal_y_regression": z.OUT / "eval-y-42-trajectory.jsonl",
            "aa_regression": aa.OUT / "eval-confirm-42-trajectory.jsonl"}


def product_config() -> dict:
    config = q.load_config()["product"]
    expected = {"retriever": "bm25-v2", "top_k": 3, "system_prompt": "direct-v090", "thinking": False,
                "max_input_tokens": 2048, "max_new_tokens": 128, "max_model_calls_per_case": 4,
                "max_searches_per_case": 3, "observation_encoding": "legacy"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Goal Q product settings do not match frozen experiment: {config}")
    return {**config, "greedy": True, "temperature": 0.0, "do_sample": False,
            "sampling_top_p": None, "sampling_top_k": None, "seed": 42,
            "seed_per_step": "seed + call_index", "timeout": None}


def prepare() -> str:
    revision = 1
    previous_manifest_sha = None
    prior_replay = None
    if OUT.exists():
        state_path = OUT / "state.json"
        state = read(state_path) if state_path.exists() else {}
        current_revision = state.get("revision", 1)
        output_exists = bool(list(OUT.glob("eval-*.jsonl")))
        if (state.get("status") not in {"awaiting_user_approval", "failed_pre_generation"}
                or current_revision not in (1, 2, 3, 4, 5, 6)
                or state.get("approved_manifest_sha256") is not None or output_exists):
            raise FileExistsError(f"refusing to overwrite or revise an active Goal AF run: {OUT}")
        previous_manifest_sha = sha(OUT / "config.json")
        if current_revision >= 2:
            old = read(OUT / "config.json").get("cpu_retrieval_replay", {})
            replay_file = OUT / "cpu-retrieval-replay.jsonl"
            if not replay_file.is_file() or sha(replay_file) != old.get("sha256"):
                raise RuntimeError("frozen CPU replay is missing or has changed")
            prior_replay = old
        archive_revision = current_revision
        for original, archived in ((OUT / "config.json", OUT / f"config-preconditions-revision-{archive_revision}.json"),
                                   (OUT / "PRECONDITIONS.md", OUT / f"PRECONDITIONS-revision-{archive_revision}.md"),
                                   (OUT / "preflight.json", OUT / f"preflight-revision-{archive_revision}.json")):
            if original.exists():
                if archived.exists():
                    if sha(original) != sha(archived):
                        raise RuntimeError(f"preflight archive differs from the original frozen file: {archived}")
                else:
                    shutil.copy2(original, archived)
        revision = current_revision + 1
    release_identity = validate_release_identity()
    cases = cohort_cases()
    corpus_hashes = {key: sha(path) for key, path in ae.original_paths().items()}
    paragraph_hashes = {key: sha(path) for key, path in ae.paragraph_paths().items()}
    historical = {key: sha(path) for key, path in historical_paths().items()}
    if any(len(ae.read_rows(path)) != COUNTS[key] for key, path in historical_paths().items()):
        raise RuntimeError("historical published-model output row counts changed")
    software = software_facts()
    inventory = files()
    cfg = {
        "version": "goal-af-retrieval-validation-v1", "stage": "awaiting_user_approval",
        "preconditions_revision": revision,
        "remote": {"ssh": "ssh -p 25320 root@connect.cqa1.seetacloud.com",
                   "repository": "/root/autodl-tmp/mini-agent-rl", "python": ".venv/bin/python"},
        "model": {"base": str(BASE_MODEL), "adapter": str(REMOTE_ADAPTER),
                  "adapter_sha256": PUBLISHED_ADAPTER_SHA256,
                  "expected_base_weight_sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                  "dtype": "bfloat16", "quantization": False, "training": False,
                  "release_identity": release_identity},
        "software": software, "product": product_config(), "retrieval_arms": ARM_RETRIEVERS,
        "repair_rules": {"page_candidates": 3, "page_index": "BM25-v2 with full-page aggregate and title boost",
            "paragraph_ranker": "body-only BM25; global paragraph IDF; BM25-v2 k1=1.5,b=0.75",
            "excluded_pure_headings": sorted(__import__("mini_agent_rl.tools.search", fromlist=["_PageParagraphBM25Index"])
                ._PageParagraphBM25Index._SECTION_HEADINGS),
            "intro_tokens": 96, "best_passage_tokens": 288, "fallback_lead_tokens": 384,
            "window": "max distinct query token coverage, then total query token matches, then earliest offset",
            "ties": "page score desc/title normalized; passage score desc/paragraph index asc/id asc",
            "returned_text": "concatenated exact source spans only; provenance retained in trace sidecar"},
        "cohorts": {key: {"n": COUNTS[key], "case_ids": [row["case_id"] for row in rows],
                          "original_corpus_sha256": corpus_hashes[key],
                          "paragraph_corpus_sha256": paragraph_hashes[key],
                          "historical_goal_z_output_sha256": historical[key]}
                     for key, rows in cases.items()},
        "total_trajectories": sum(COUNTS.values()) * 3,
        "max_model_calls": sum(COUNTS.values()) * 3 * 4,
        "max_searches": sum(COUNTS.values()) * 3 * 3,
        "baseline_expected": {"goal_z_dev_correct_sufficient": 36, "goal_z_dev_bad_total": 19,
            "goal_z_dev_invalid": 0, "old_main": 15, "old_regression": 12,
            "goal_y": 40, "aa_regression": 32},
        "gate": {"dev_success_gain_min": 4, "dev_bad_not_increase": True,
            "dev_invalid_not_increase": True, "each_regression_max_loss": 1,
            "aa_multihop_protected_max_loss": 1,
            "ranking": ["dev_success", "fewer_dev_bad", "fewer_mean_searches", "simpler_arm_B"]},
        "confirmation": {"opened": False, "run_this_goal": False,
            "manifest_hash_only": release_identity["fresh_confirmation_case_and_corpus_sha256"]},
        "inputs": {name: sha(path) for name, path in inventory.items()},
        "review_rule_sha256": sha(z.OUT / "evidence-rubric.json"),
        "approval_required": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cfg["cpu_retrieval_replay"] = prior_replay or replay_saved_queries()
    config_path = OUT / "config.json"
    if config_path.exists():
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        write_once(config_path, cfg)
    manifest_sha = sha(config_path)
    state_value = {"status": "awaiting_user_approval", "approved_manifest_sha256": None,
                   "completed_arms": [], "confirmation_accessed": False, "revision": revision,
                   "supersedes_manifest_sha256": previous_manifest_sha}
    if (OUT / "state.json").exists():
        write_state(OUT / "state.json", state_value)
    else:
        write_once(OUT / "state.json", state_value)
    preconditions_path = OUT / "PRECONDITIONS.md"
    preconditions_text = render_preconditions(cfg, manifest_sha)
    if preconditions_path.exists():
        preconditions_path.write_text(preconditions_text, encoding="utf-8")
    else:
        write_once(preconditions_path, preconditions_text)
    preflight = {"manifest_sha256": manifest_sha, "identity": release_identity,
        "software": software, "cases": COUNTS, "fresh_confirmation_opened": False,
        "baseline_references": {key: str(value) for key, value in historical_paths().items()},
        "cpu_retrieval_replay": cfg["cpu_retrieval_replay"]}
    preflight_path = OUT / "preflight.json"
    if preflight_path.exists():
        preflight_path.write_text(json.dumps(preflight, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        write_once(preflight_path, preflight)
    print(json.dumps({"status": "awaiting_user_approval", "manifest_sha256": manifest_sha,
                      "preconditions": str(OUT / "PRECONDITIONS.md")}, ensure_ascii=False))
    return manifest_sha


def render_preconditions(cfg: dict, manifest_sha: str) -> str:
    lines = ["# Goal AF 实验前提", "", f"配置 SHA-256：`{manifest_sha}`", "",
        "GPU 输出尚未生成。运行前需要用户确认以上哈希对应的完整清单。", "",
        "## 模型与运行环境", "",
        f"- 基础模型：`{cfg['model']['base']}`；BF16，单卡，不量化，不训练。",
        f"- 已发布 adapter：`{cfg['model']['adapter']}`",
        f"- adapter SHA-256：`{cfg['model']['adapter_sha256']}`",
        f"- 基础模型实际权重 SHA-256：`{json.dumps(cfg['model']['release_identity']['base_weight_sha256'], ensure_ascii=False)}`",
        f"- Tokenizer 文件 SHA-256：`{cfg['inputs']['tokenizer']}`；tokenizer 关联文件均在 `config.json` inputs 中逐项哈希。", "",
        "## 评价与搜索设置", "",
        "- 三臂 A/B/C：BM25-v2 原语料、冻结 BM25-v3 段落方案、冻结 BM25-v4 页面定位加正文选段。",
        "- Goal Q 入口：direct-v090、thinking=false、legacy；top-3；每题最多4次模型调用/3次搜索。",
        "- Greedy：temperature=0、do_sample=false；每次最多2048输入/128新 token。",
        "- 四组固定问题共208题，每臂一次，共624条轨迹；历史确认集只用于AA回归评价。", "",
        "## 数据、门槛与隐私边界", "",
        "- 各组数据/原文语料/段落语料/历史输出哈希和题目ID均冻结在 `config.json`。",
        "- 先完成 A 并逐题复核 Goal Z 历史基线；不一致时停止，不执行 B/C。",
        "- 使用冻结证据规则评分实际回答时可见内容；新确认集保持关闭，仅核验哈希。",
        "- 候选开发提升至少4题，开发错误与无效动作不增，四组回归各最多下降1题，AA保护题最多丢失1题。", "",
        "## 前提核查值", "",
        f"- 软件：Python {cfg['software']['python']}、torch {cfg['software']['torch']}、transformers {cfg['software']['transformers']}、PEFT {cfg['software']['peft']}、tokenizers {cfg['software']['tokenizers']}、accelerate {cfg['software']['accelerate']}、CUDA {cfg['software']['cuda_runtime']}。",
        f"- GPU 与驱动：{cfg['software']['gpu'] or '未检测到'}；{cfg['software']['nvidia_smi_gpu_driver_memory'] or 'nvidia-smi 未提供驱动信息'}。", "",
        "执行接口：前提清单冻结后运行 `run`；配置哈希记录用于追溯，模型与数据资产仍按清单校验。"]
    return "\n".join(lines) + "\n"


def config() -> dict:
    path = OUT / "config.json"
    cfg = read(path)
    current = {name: sha(file) for name, file in files().items()}
    if current != cfg["inputs"]:
        changed = sorted(name for name in set(current) | set(cfg["inputs"]) if current.get(name) != cfg["inputs"].get(name))
        raise RuntimeError(f"frozen inputs changed: {changed}")
    validate_release_identity()
    if software_facts() != cfg["software"]:
        raise RuntimeError("runtime, CUDA, GPU, or software versions changed after prepare")
    return cfg


def verify_loaded_adapter(client) -> dict:
    import torch
    from safetensors.torch import load_file
    expected = load_file(str(PUBLISHED_ADAPTER / "adapter_model.safetensors"), device="cpu")
    # PEFT's ordinary state_dict adds the active adapter name (for example,
    # ``lora_A.default.weight``), while adapter_model.safetensors omits it.
    # Compare the canonical adapter-only state returned by PEFT instead.
    loaded = peft_adapter_state(client._model)
    missing = sorted(set(expected) - set(loaded))
    if missing:
        raise RuntimeError(f"loaded adapter keys missing from active model: {missing[:8]}")
    mismatched = [key for key, tensor in expected.items() if not torch.equal(loaded[key].detach().cpu(), tensor)]
    if mismatched:
        raise RuntimeError(f"loaded adapter tensors differ from pinned adapter: {mismatched[:8]}")
    active = getattr(client._model, "active_adapters", getattr(client._model, "active_adapter", None))
    if active not in ("default", ["default"]):
        raise RuntimeError(f"unexpected active PEFT adapter: {active}")
    peft_cfg = client._model.peft_config["default"]
    if not base_model_path_matches(peft_cfg.base_model_name_or_path, BASE_MODEL):
        raise RuntimeError(f"adapter was prepared for another base model: {peft_cfg.base_model_name_or_path}")
    return {"adapter_sha256": sha(PUBLISHED_ADAPTER / "adapter_model.safetensors"),
            "loaded_adapter_tensor_count": len(expected), "active_adapter": active,
            "peft_base_model": str(peft_cfg.base_model_name_or_path)}


def peft_adapter_state(model) -> dict:
    from peft import get_peft_model_state_dict
    return get_peft_model_state_dict(model, adapter_name="default")


def load_and_verify_adapter(client) -> dict:
    """Materialize the lazy model before checking its active adapter tensors."""
    if client._model is None:
        client._load()
    if client._model is None:
        raise RuntimeError("model loader returned without materializing model weights")
    return verify_loaded_adapter(client)


def compare_historical(current_rows: list[dict], historical_rows: list[dict]) -> dict:
    left = {row["case_id"]: row for row in current_rows}
    right = {row["case_id"]: row for row in historical_rows}
    if set(left) != set(right):
        return {"passed": False, "reason": "case_id set differs",
                "current_only": sorted(set(left) - set(right)), "historical_only": sorted(set(right) - set(left))}
    differences = []
    for case_id in sorted(left):
        a, b = left[case_id], right[case_id]
        fields = ("answer", "reason", "searches", "model_calls")
        if any(a.get(key) != b.get(key) for key in fields):
            differences.append({"case_id": case_id, "fields": [key for key in fields if a.get(key) != b.get(key)]})
            continue
        if len(a.get("trace", [])) != len(b.get("trace", [])):
            differences.append({"case_id": case_id, "fields": ["trace_length"]})
            continue
        for ix, (ta, tb) in enumerate(zip(a["trace"], b["trace"])):
            if (ta.get("prompt_token_ids") != tb.get("prompt_token_ids") or ta.get("query") != tb.get("query")
                    or ta.get("action") != tb.get("action") or ta.get("answer") != tb.get("answer")):
                differences.append({"case_id": case_id, "step": ix, "fields": ["prompt_or_action"]})
                break
            oa, ob = ta.get("observation", {}).get("result"), tb.get("observation", {}).get("result")
            if oa != ob:
                differences.append({"case_id": case_id, "step": ix, "fields": ["tool_results"]})
                break
    return {"passed": not differences, "difference_count": len(differences), "differences": differences}


def evaluate_arm(arm: str, manifest_sha: str) -> dict:
    cfg = config()
    state_path = OUT / "state.json"
    state = read(state_path)
    if state["status"] not in {"running", "baseline_verified"}:
        raise RuntimeError(f"cannot run evaluation from state {state['status']}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; preserving state without restarting")
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    from run_v133_goal_p_variant import SearchTextAliasClient
    from transformers import AutoTokenizer

    product = cfg["product"]
    client = TransformersModelClient(LocalModelConfig(model_id=str(BASE_MODEL), adapter_path=str(PUBLISHED_ADAPTER),
        dtype="bfloat16", load_in_4bit=False, max_input_tokens=2048, max_new_tokens=128,
        timeout=None, enable_thinking=False, deterministic=True, observation_encoding="legacy",
        recompute_old_logprobs=False))
    # Verify the active adapter before any generation is allowed.
    loaded_identity = load_and_verify_adapter(client)
    wrapped = SearchTextAliasClient(client)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True, trust_remote_code=False)
    cases = cohort_cases()
    output_path = OUT / f"eval-{arm}.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.monotonic()
    completed = 0
    for split in SPLITS:
        corpus_path = ae.original_paths()[split] if arm == "A" else ae.paragraph_paths()[split]
        docs = read(corpus_path)
        registry = ToolRegistry()
        tool = SearchTool(docs, ARM_RETRIEVERS[arm], tokenizer=tokenizer if arm == "C" else None)
        registry.register(tool)
        for row in cases[split]:
            if ARM_RETRIEVERS[arm] != product["retriever"] and arm == "A":
                raise RuntimeError("A retrieval configuration drifted from Goal Q")
            if hasattr(tool._bm25_index, "source_history"):
                tool._bm25_index.source_history.clear()
            result = asyncio.run(q.rollout(row["question"], wrapped, registry, sample=False, seed=42))
            result.update(case_id=row["case_id"], split=split, arm=arm, model="published_goal_z_trajectory_seed42",
                model_identity=loaded_identity, manifest_sha256=manifest_sha,
                support_titles=row["support_titles"],
                support_retrieved=bool(row["support_titles"]) and set(row["support_titles"]).issubset(
                    {str(doc.get("title", "")) for step in result["trace"]
                     for doc in step.get("observation", {}).get("result", [])}),
                source_provenance=(tool._bm25_index.source_history if arm == "C" else []),
                prompt_texts=[tokenizer.decode(step["prompt_token_ids"], skip_special_tokens=False)
                              for step in result["trace"]])
            with output_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
            completed += 1
            if completed % 8 == 0:
                print(json.dumps({"arm": arm, "completed": completed, "total": sum(COUNTS.values()),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                    "peak_allocated_gb": client.gpu_memory_peak_gb()}, ensure_ascii=False), flush=True)
    del client, wrapped
    torch.cuda.empty_cache()
    if completed != sum(COUNTS.values()):
        raise RuntimeError(f"incomplete {arm} evaluation: {completed}")
    result = {"arm": arm, "rows": completed, "sha256": sha(output_path),
              "seconds": time.monotonic() - started}
    state = read(state_path)
    state["arms"][arm] = result
    state["completed_trajectories"] = sum(item["rows"] for item in state["arms"].values())
    if arm == "A":
        current = ae.read_rows(output_path)
        old = []
        for split, path in historical_paths().items():
            old.extend(ae.read_rows(path))
        comparison = compare_historical(current, old)
        state["baseline_comparison"] = comparison
        if not comparison["passed"]:
            state["status"] = "baseline_mismatch_stop"
            write_state(state_path, state)
            raise RuntimeError("A baseline differs from published Goal Z outputs; stopping before B/C")
        state["status"] = "baseline_verified"
    else:
        state["status"] = "running"
    write_state(state_path, state)
    return result


def write_state(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def attach_em_f1(rows: list[dict], gold_by_case: dict[str, str]) -> list[dict]:
    """Recompute answer scores from immutable outputs and frozen case labels."""
    scored = []
    for row in rows:
        if row["case_id"] not in gold_by_case:
            raise RuntimeError(f"evaluation output has unknown case_id: {row['case_id']}")
        scored.append({**row, **q.score(row.get("answer"), gold_by_case[row["case_id"]])})
    return scored


def summarize() -> dict:
    if not (OUT / "config.json").exists():
        raise FileNotFoundError("run prepare first")
    cfg = read(OUT / "config.json")
    arms = {}
    review_by_hash = {}
    review_mapping = []
    sub_split_by_id = {case["case_id"]: case.get("subsplit")
                       for cases in cohort_cases().values() for case in cases}
    for arm in ARM_RETRIEVERS:
        path = OUT / f"eval-{arm}.jsonl"
        if not path.exists():
            arms[arm] = {"status": "not_run"}
            continue
        rows = ae.read_rows(path)
        gold_by_case = {case["case_id"]: case["answer"]
                        for cohort in cohort_cases().values() for case in cohort}
        rows = attach_em_f1(rows, gold_by_case)
        by_split = {}
        for split in SPLITS:
            part = [row for row in rows if row["split"] == split]
            by_split[split] = {"n": len(part), "em": sum(row["em"] for row in part),
                "mean_f1": statistics.mean(row["f1"] for row in part) if part else 0.0,
                "invalid_actions": sum(row["reason"] == "invalid_action" for row in part),
                "unanswered": sum(not row.get("answer") for row in part),
                "support_title_hits": sum(row["support_retrieved"] for row in part),
                "mean_searches": statistics.mean(row["searches"] for row in part) if part else 0.0,
                "mean_latency_seconds": statistics.mean(row["elapsed_seconds"] for row in part) if part else 0.0}
        for row in rows:
            signature = digest({"question": row["question"],
                "trace": [{"prompt": prompt, "action": trace.get("action"), "query": trace.get("query"),
                    "answer": trace.get("answer"), "observation": trace.get("observation", {}).get("result")}
                    for prompt, trace in zip(row["prompt_texts"], row["trace"])],
                "answer": row.get("answer")})
            review_by_hash.setdefault(signature, {"review_id": signature, "case_id": row["case_id"],
                "question": row["question"], "calls": [{"actual_prompt": prompt,
                    "action": trace.get("action"), "query": trace.get("query"), "answer": trace.get("answer"),
                    "actual_tool_results": trace.get("observation", {}).get("result")}
                    for prompt, trace in zip(row["prompt_texts"], row["trace"])],
                "answer": row.get("answer"), "arm_labels_hidden": True})
            review_mapping.append({"review_id": signature, "arm": arm, "split": row["split"],
                "case_id": row["case_id"], "subsplit": sub_split_by_id.get(row["case_id"])})
        arms[arm] = {"status": "awaiting_semantic_review", "n": len(rows), "by_split": by_split,
                     "em_total": sum(row["em"] for row in rows), "output_sha256": sha(path)}
    packet = OUT / "anonymous-semantic-review-packet.jsonl"
    if not packet.exists() and review_by_hash:
        packet.parent.mkdir(parents=True, exist_ok=True)
        with packet.open("x", encoding="utf-8", newline="\n") as stream:
            for value in review_by_hash.values():
                stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    map_path = OUT / "semantic-review-map.jsonl"
    if not map_path.exists() and review_mapping:
        with map_path.open("x", encoding="utf-8", newline="\n") as stream:
            for value in review_mapping:
                stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    result = {"version": "goal-af-summary-v1", "arms": arms,
        "semantic_review_packet_rows": len(review_by_hash), "confirmation_accessed": False,
        "product_change_published": False, "output_count": sum(arm.get("n", 0) for arm in arms.values())}
    review_path = OUT / "semantic-reviews.jsonl"
    full_eval = all(arms.get(arm, {}).get("n") == sum(COUNTS.values()) for arm in ARM_RETRIEVERS)
    if review_path.exists() and full_eval:
        review = validate_semantic_reviews(review_by_hash, ae.read_rows(review_path))
        if review["complete"]:
            result["semantic"] = apply_semantic_gate(review, review_mapping,
                {arm: ae.read_rows(OUT / f"eval-{arm}.jsonl") for arm in ARM_RETRIEVERS}, cfg)
            result["status"] = result["semantic"]["decision"]
        else:
            result["status"] = "awaiting_semantic_review"
            result["semantic_review_remaining"] = review["missing_count"]
    else:
        result["status"] = "awaiting_semantic_review" if review_by_hash else "evaluation_incomplete"
    summary_path = OUT / ("summary.json" if result["status"] != "awaiting_semantic_review" else "summary-pending-review.json")
    if summary_path.exists():
        raise FileExistsError("summary already exists; summaries are immutable")
    summary_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report_path = OUT / ("REPORT.md" if result["status"] != "awaiting_semantic_review" else "REPORT-pending-review.md")
    report_path.write_text(render_summary(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def render_summary(result: dict) -> str:
    lines = ["# Goal AF 当前结果", "", "本报告不打开新确认集，不自动发布检索配置。", "",
             "| Arm | N | EM | Z开发 | 旧题 | Goal Y | AA回归 | 平均F1 | 平均搜索 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, values in result["arms"].items():
        if values.get("status") == "not_run":
            lines.append(f"| {arm} | — | — | — | — | — | — | — | — |")
            continue
        splits = values["by_split"]
        vals = [splits[key]["em"] for key in SPLITS]
        gold_by_case = {case["case_id"]: case["answer"]
                        for cohort in cohort_cases().values() for case in cohort}
        scored_rows = attach_em_f1(ae.read_rows(OUT / f"eval-{arm}.jsonl"), gold_by_case)
        f1 = statistics.mean(row["f1"] for row in scored_rows)
        searches = statistics.mean(row["searches"] for row in scored_rows)
        lines.append(f"| {arm} | {values['n']} | {values['em_total']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {f1:.3f} | {searches:.2f} |")
    lines += ["", f"状态：{result.get('status', 'awaiting_semantic_review')}。新确认集保持关闭，产品检索配置未切换。",
              "语义审核使用 `anonymous-semantic-review-packet.jsonl`；完成后按一行一个 JSON 追加到 `semantic-reviews.jsonl`。",
              "EM、支持标题命中或搜索次数都不代替实际可见证据判断。", ""]
    if result.get("semantic"):
        lines.append(f"开发决策：{result['semantic']['decision']}；候选：{result['semantic'].get('selected_arm') or '无'}。")
    return "\n".join(lines)


def validate_semantic_reviews(packet_by_id: dict, reviews: list[dict]) -> dict:
    categories = {"correct_sufficient", "missing_required_information", "factual_error",
                  "unsupported_by_visible_evidence", "indeterminate"}
    found = {}
    for row in reviews:
        review_id = row.get("review_id")
        if review_id not in packet_by_id or review_id in found or row.get("category") not in categories:
            raise RuntimeError("semantic review contains an unknown, duplicate, or invalid review id/category")
        packet = packet_by_id[review_id]
        if not row.get("reason"):
            raise RuntimeError(f"semantic review lacks rationale: {review_id}")
        if not isinstance(row.get("diagnostics"), list) or not set(row["diagnostics"]).issubset(DIAGNOSTICS):
            raise RuntimeError(f"semantic review has missing or unknown trajectory diagnostics: {review_id}")
        for quote in row.get("evidence_quotes", []):
            call_index = quote.get("call_index")
            source_id, text = str(quote.get("source_id", "")), str(quote.get("text", ""))
            if not isinstance(call_index, int) or not 0 <= call_index < len(packet["calls"]):
                raise RuntimeError(f"evidence citation points outside the actual trace: {review_id}")
            visible = packet["calls"][call_index].get("actual_tool_results") or []
            if not any(str(doc.get("id")) == source_id and text and text in str(doc.get("text", "")) for doc in visible):
                raise RuntimeError(f"evidence citation is not present in actual model-visible results: {review_id}")
            final_prompt = packet["calls"][-1].get("actual_prompt", "") if packet["calls"] else ""
            if text not in final_prompt:
                raise RuntimeError(f"cited evidence was truncated before the final answer input: {review_id}")
        sufficient_call = row.get("first_sufficient_call")
        if row["category"] == "correct_sufficient":
            if not row.get("evidence_quotes") or not isinstance(sufficient_call, int) or \
               not 0 <= sufficient_call < len(packet["calls"]):
                raise RuntimeError(f"success requires visible citations and the first sufficient call: {review_id}")
        elif sufficient_call is not None and (not isinstance(sufficient_call, int)
                or not 0 <= sufficient_call < len(packet["calls"])):
            raise RuntimeError(f"invalid first sufficient call: {review_id}")
        found[review_id] = row
    missing = sorted(set(packet_by_id) - set(found))
    return {"complete": not missing, "missing_count": len(missing), "missing_ids": missing, "by_id": found}


def apply_semantic_gate(review: dict, mapping: list[dict], rows_by_arm: dict, cfg: dict) -> dict:
    votes = review["by_id"]
    by_case = {(item["arm"], item["split"], item["case_id"]): votes[item["review_id"]]
               for item in mapping}
    raw = {(arm, row["split"], row["case_id"]): row
           for arm, values in rows_by_arm.items() for row in values}
    subsplits = {(item["arm"], item["case_id"]): item.get("subsplit") for item in mapping}

    def counts(arm: str, split: str, subsplit: str | None = None) -> dict:
        keys = [key for key in by_case if key[0] == arm and key[1] == split
                and (subsplit is None or subsplits.get((arm, key[2])) == subsplit)]
        categories = [by_case[key]["category"] for key in keys]
        return {"n": len(categories), "correct_sufficient": categories.count("correct_sufficient"),
            "bad_total": categories.count("factual_error") + categories.count("unsupported_by_visible_evidence"),
            "factual_error": categories.count("factual_error"),
            "unsupported_by_visible_evidence": categories.count("unsupported_by_visible_evidence"),
            "invalid": sum(raw[key].get("reason") == "invalid_action" for key in keys)}

    baseline = {"goal_z_dev": counts("A", "goal_z_dev"),
        "old_main": counts("A", "old_regression", "new_eval"),
        "old_regression": counts("A", "old_regression", "goal_G_regression"),
        "goal_y_regression": counts("A", "goal_y_regression"),
        "aa_regression": counts("A", "aa_regression")}
    diagnostic_counts = {}
    for arm in ARM_RETRIEVERS:
        for split in SPLITS:
            items = [item for item in mapping if item["arm"] == arm and item["split"] == split]
            diagnostic_counts[f"{arm}-{split}"] = {event: sum(event in by_case[item["arm"], item["split"], item["case_id"]].get("diagnostics", [])
                for item in items) for event in sorted(DIAGNOSTICS)}
    expected = cfg["baseline_expected"]
    baseline_matches = (baseline["goal_z_dev"]["correct_sufficient"] == expected["goal_z_dev_correct_sufficient"]
        and baseline["goal_z_dev"]["bad_total"] == expected["goal_z_dev_bad_total"]
        and baseline["goal_z_dev"]["invalid"] == expected["goal_z_dev_invalid"]
        and baseline["old_main"]["correct_sufficient"] == expected["old_main"]
        and baseline["old_regression"]["correct_sufficient"] == expected["old_regression"]
        and baseline["goal_y_regression"]["correct_sufficient"] == expected["goal_y"]
        and baseline["aa_regression"]["correct_sufficient"] == expected["aa_regression"])
    protected = {key[2] for key, vote in by_case.items()
        if key[0] == "A" and key[1] == "aa_regression" and vote["category"] == "correct_sufficient"
        and vote["first_sufficient_call"] > 0}
    candidates = {}
    for arm in ("B", "C"):
        values = {"goal_z_dev": counts(arm, "goal_z_dev"),
            "old_main": counts(arm, "old_regression", "new_eval"),
            "old_regression": counts(arm, "old_regression", "goal_G_regression"),
            "goal_y_regression": counts(arm, "goal_y_regression"),
            "aa_regression": counts(arm, "aa_regression")}
        lost = sum(by_case[arm, "aa_regression", case_id]["category"] != "correct_sufficient" for case_id in protected)
        values["protected_multihop_count"] = len(protected)
        values["protected_multihop_lost"] = lost
        values["mean_searches"] = statistics.mean(row["searches"] for row in rows_by_arm[arm])
        values["passed"] = (baseline_matches and
            values["goal_z_dev"]["correct_sufficient"] >= baseline["goal_z_dev"]["correct_sufficient"] + 4 and
            values["goal_z_dev"]["bad_total"] <= baseline["goal_z_dev"]["bad_total"] and
            values["goal_z_dev"]["invalid"] <= baseline["goal_z_dev"]["invalid"] and
            all(values[group]["correct_sufficient"] >= baseline[group]["correct_sufficient"] - 1
                for group in ("old_main", "old_regression", "goal_y_regression", "aa_regression")) and
            lost <= 1 and all(values[group]["n"] == baseline[group]["n"] for group in baseline))
        candidates[arm] = values
    eligible = [arm for arm, value in candidates.items() if value["passed"]]
    eligible.sort(key=lambda arm: (-candidates[arm]["goal_z_dev"]["correct_sufficient"],
        candidates[arm]["goal_z_dev"]["bad_total"], candidates[arm]["mean_searches"], 0 if arm == "B" else 1))
    return {"baseline": baseline, "baseline_matches_frozen_expectation": baseline_matches,
        "trajectory_diagnostic_counts": diagnostic_counts,
        "protected_multihop_case_ids": sorted(protected), "candidates": candidates,
        "eligible_arms": eligible, "selected_arm": eligible[0] if eligible else None,
        "decision": "candidate_eligible_for_separate_confirmation_plan" if eligible else "no_candidate_passed_development_gate",
        "product_change_published": False, "fresh_confirmation_opened": False}


def run(approved_manifest_sha256: str | None = None) -> None:
    cfg = config()
    config_hash = sha(OUT / "config.json")
    if approved_manifest_sha256 is not None and approved_manifest_sha256 != config_hash:
        raise RuntimeError("provided hash does not match frozen preconditions")
    state_path = OUT / "state.json"
    state = read(state_path)
    if state["status"] != "awaiting_user_approval":
        raise RuntimeError(f"run cannot start from state {state['status']}")
    state.update(status="running", approved_manifest_sha256=config_hash, arms={})
    write_state(state_path, state)
    for arm in ("A", "B", "C"):
        evaluate_arm(arm, config_hash)
        state = read(state_path)
        state["status"] = "running"
        write_state(state_path, state)
    state = read(state_path)
    state["status"] = "evaluation_complete"
    write_state(state_path, state)
    summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--approved-manifest-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "summarize":
        summarize()
    else:
        run(args.approved_manifest_sha256)


if __name__ == "__main__":
    main()
